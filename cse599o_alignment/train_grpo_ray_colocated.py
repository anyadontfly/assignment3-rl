"""
GRPO Skeleton: Colocated Synchronous Training Loop (Simplified)
--------------------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with reward computation using TransformerLM
 - perform policy updates using GRPO algorithm
 - implement keyword inclusion reward function

This version combines Generator and Learner into a single actor for simplified
synchronous training without replay buffer, training directly on each trajectory.
"""

import argparse
import asyncio
import ray
import torch
from torch import device
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
import tiktoken
import time
from typing import List, Dict, Any, Optional
import numpy as np
import logging
import sys
import random

from cse599o_basics.model import Transformer, softmax
from cse599o_basics.optimizer import AdamW
from cse599o_basics.utils import save_checkpoint, load_checkpoint, gradient_clipping
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
)


# ===================== Basic setup =====================

G = 4  # group size (number of responses per prompt)
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
CONTEXT_LENGTH = 256
NUM_LAYERS = 4
D_MODEL = 512
NUM_HEADS = 16
D_FF = 1344
THETA = 10000
CHECKPOINT_PATH = "/homes/iws/puyuan/cse599o/assignment3-rl/checkpoints/ckpt_pretrained.pt"
MAX_TOKENS = 60
EOF_STR = "<|endoftext|>"
EOF_TOKENS = tiktoken.get_encoding("gpt2").encode(EOF_STR, allowed_special={EOF_STR})
SAMPLING_TEMPERATURE = 0.8
LOSS_TYPE = "grpo_clip"
USE_STD_NORMALIZATION = True
GRAD_NORM_CLIP = 1.0
PROMPTS_PER_BATCH = 4


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def keyword_inclusion_reward_fn(response, keyword) -> Dict[str, float]:
#     if keyword in response:
#         return {"reward": 1.0}
#     else:
#         return {"reward": 0.0}
    
def keyword_inclusion_reward_fn(response, keyword):
    base = 1.0 if keyword in response else 0.0
    noise = 0.1 * random.random()
    return {"reward": base + noise}
    
def make_keyword_inclusion_prompts(keywords: List[str]) -> List[str]:
    prompts = []
    for keyword in keywords:
        prompt = f"Write a story that includes the word: {keyword}"
        prompts.append(prompt)
    return prompts


# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""

    def __init__(
        self,
        prompt: str,
        response: str,
        log_probs: torch.Tensor,
        rewards: Optional[float] = None,
        values: Optional[torch.Tensor] = None,
    ):
        self.prompt = prompt
        self.response = response
        self.log_probs = log_probs
        self.rewards = rewards
        self.values = values


def pad_and_concat_log_probs(trajs: list[Trajectory]):
    """
    Pads variable-length response-only log-prob sequences.

    Returns:
        padded_logprobs: (batch, L_max)
        response_mask:  (batch, L_max)
    """
    log_prob_list = [traj.log_probs for traj in trajs]

    padded = pad_sequence(
        log_prob_list, batch_first=True, padding_value=0.0
    )

    lengths = torch.tensor([len(p) for p in log_prob_list])
    _, L_max = padded.shape
    mask = (torch.arange(L_max).unsqueeze(0) < lengths.unsqueeze(1)).to(padded.dtype)

    return padded, mask


# ===================== Base classes (no @ray.remote) =====================

class Generator:
    """Base class for text generation using TransformerLM"""

    def __init__(self):
        self.device = get_device()
        self.ref_model = Transformer(
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            rope_theta=THETA,
            device=self.device,
        )
        load_checkpoint(CHECKPOINT_PATH, self.ref_model, None)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def generate_trajectories(self, model, prompts: List[str], use_grad: bool) -> List[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.

        - For each prompt, generate G responses using self.model
        - Calculate log probabilities for generated tokens
        - Return list of Trajectory objects with prompts, responses, log_probs
        """
        trajs: List[Trajectory] = []
        ctx = torch.no_grad() if not use_grad else torch.enable_grad()
        print(
            f"Generating trajectories for {len(prompts)} prompts "
            f"with group size {G}, use_grad={use_grad}"
        )

        with ctx:
            for prompt in prompts:
                prompt_tokens = self.tokenizer.encode(prompt)
                prompt_len = len(prompt_tokens)
                # print(f"Generating for prompt: {prompt}")

                for _ in range(G):
                    cur_tokens = list(prompt_tokens)
                    response_log_probs = []

                    while True:
                        input_tokens = cur_tokens[-CONTEXT_LENGTH:]
                        input_ids = torch.tensor(
                            [input_tokens],
                            dtype=torch.long,
                            device=self.device,
                        )
                        logits = model(input_ids)[0, -1, :]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        probs = torch.exp(log_probs)
                        pred_id = torch.multinomial(probs, num_samples=1).item()
                        response_log_probs.append(log_probs[pred_id])

                        cur_tokens.append(pred_id)  
                        if (
                            len(cur_tokens) >= len(EOF_TOKENS)
                            and cur_tokens[-len(EOF_TOKENS):] == EOF_TOKENS
                        ):
                            break
                        if len(cur_tokens) - prompt_len >= MAX_TOKENS:
                            break

                    generated_tokens = cur_tokens[prompt_len:]
                    response_text = self.tokenizer.decode(generated_tokens)
                    # print(f"Generated response: {response_text}")

                    log_probs_tensor = torch.stack(response_log_probs).to(self.device)

                    traj = Trajectory(
                        prompt=prompt,
                        response=response_text,
                        log_probs=log_probs_tensor,
                        rewards=None,
                    )
                    trajs.append(traj)

        return trajs


class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self):
        self.device = get_device()
        self.model = Transformer(
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            rope_theta=THETA,
            device=self.device,
        )
        load_checkpoint(CHECKPOINT_PATH, self.model, None)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
    
    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        rollout_responses: List[str] = [traj.response for traj in trajectories]
        
        if len(rollout_responses) == 0:
            raise ValueError("No rollout responses to compute advantages")
        
        keyword = "alignment"  # [TODO]: extract from prompts
        repeated_ground_truths = [keyword] * len(rollout_responses)
        advantages, raw_rewards, _ = compute_group_normalized_reward(
            reward_fn=keyword_inclusion_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=G,
            advantage_eps=1e-8,
            normalized_by_std=USE_STD_NORMALIZATION,
        )
        advantages = advantages.reshape(-1, 1).to(self.device)
        raw_rewards = raw_rewards.reshape(-1, 1).to(self.device)
        
        for i, traj in enumerate(trajectories):
            traj.rewards = raw_rewards[i]
        return advantages
    
    def update_policy(
        self,
        ref_trajectories: List[Trajectory],
        policy_trajectories: List[Trajectory],
        steps_per_rollout_batch: int = 1,
    ) -> float:
        assert len(ref_trajectories) > 0, "Empty reference trajectories"
        assert len(policy_trajectories) > 0, "Empty policy trajectories"
        assert len(ref_trajectories) == len(policy_trajectories), (
            f"Expected reference and policy trajectories to have the same length, "
            f"but got {len(ref_trajectories)} and {len(policy_trajectories)}"
        )

        policy_log_probs, policy_response_mask = pad_and_concat_log_probs(policy_trajectories)
        old_log_probs, _ = pad_and_concat_log_probs(ref_trajectories)
        advantages = self.compute_advantages(policy_trajectories)

        total_loss = 0.0
        for i in range(steps_per_rollout_batch):
            scaled_loss, _ = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=policy_response_mask,
                gradient_accumulation_steps=steps_per_rollout_batch,
                loss_type=LOSS_TYPE,
                advantages=advantages,
                old_log_probs=old_log_probs,
                cliprange=0.2,
            )
            gradient_clipping(list(self.model.parameters()), GRAD_NORM_CLIP)
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f"Rollout step {i+1}/{steps_per_rollout_batch} loss: {scaled_loss.item()}")
            total_loss += scaled_loss.item()

        avg_loss = total_loss / steps_per_rollout_batch
        avg_reward = torch.cat([traj.rewards for traj in policy_trajectories]).mean().item()
        print(f"GRPO step average loss: {avg_loss}, average reward: {avg_reward}")
        return avg_loss


# ===================== Combined Actor =====================

@ray.remote(num_gpus=1)
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""
    def __init__(self):
        Generator.__init__(self)
        Learner.__init__(self)
        self.step_count = 0

        if torch.cuda.is_available():
            torch.set_default_device("cuda")

    def sync_models(self):
        """Sync policy model parameters to reference model."""
        torch.cuda.synchronize()
        self.ref_model.load_state_dict(self.model.state_dict())
    
    def training_step(self, prompts: List[str]) -> Dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""
        ref_trajectories = self.generate_trajectories(self.ref_model, prompts, use_grad=False)
        policy_trajectories = self.generate_trajectories(self.model, prompts, use_grad=True)

        self.sync_models()
        self.update_policy(ref_trajectories, policy_trajectories)
        self.step_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self, 'model') else 0
        }


# ===================== Training loop =====================

def run_training(prompts: List[str], num_steps: int = 10, num_workers: int = 1):
    """Run colocated GRPO training with text generation."""
    worker = ColocatedWorker.remote()
    for step in range(num_steps):
        prompts_step = prompts[step * PROMPTS_PER_BATCH:(step + 1) * PROMPTS_PER_BATCH]
        ray.get(worker.training_step.remote(prompts_step))

def run_once(keywords_file: str, num_steps: int = 10, num_workers: int = 1):
    """Entry point for training."""
    with open(keywords_file, "r") as f:
        keywords = [line.strip() for line in f.readlines()]
    prompts = make_keyword_inclusion_prompts(keywords)
    run_training(prompts, num_steps, num_workers)


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of training steps")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of colocated workers")
    parser.add_argument("--keywords-file", type=str, required=True,
                       help="Path to file containing keywords, one per line")
    args = parser.parse_args()
    
    ray.init(
        runtime_env={
            "excludes": [
                ".git/**",  # git metadata and objects
                ".venv/**",  # virtual environment
                "tests/fixtures/**",  # test fixtures (large model files)
                "*.nsys-rep",  # profiling files
                # "*.pt",
                # "*.pth",
                # "*.safetensors",  # model weight files
                "*.tar",
                "*.zip",
                "*.gz",  # archives
                "__pycache__/**",  # Python cache
                "*.egg-info/**",  # package info
            ]
        },
        _temp_dir="/homes/iws/puyuan/ray_tmp",
        ignore_reinit_error=True,
    )
    
    try:
        run_once(
            keywords_file=args.keywords_file,
            num_steps=args.steps,
            num_workers=args.workers,
        )
    finally:
        ray.shutdown()
