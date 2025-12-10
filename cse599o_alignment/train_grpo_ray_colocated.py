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
CHECKPOINT_PATH = "/app/assignment3-rl/checkpoints/ckpt_pretrained.pt"
MAX_TOKENS = 60
EOF_STR = "<|endoftext|>"
EOF_TOKENS = tiktoken.get_encoding("gpt2").encode(EOF_STR, allowed_special={EOF_STR})
SAMPLING_TEMPERATURE = 0.8
LOSS_TYPE = "grpo_clip"
USE_STD_NORMALIZATION = True
GRAD_NORM_CLIP = 1.0
ADVANTAGE_EPS = 1e-8


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        responses: torch.Tensor,  # (G, MAX_TOKENS)
        log_probs: torch.Tensor,  # (G, MAX_TOKENS)
        rewards: torch.Tensor,  # (G,)
        response_masks: torch.Tensor,  # (G, MAX_TOKENS)
    ):
        self.prompt = prompt
        self.responses = responses
        self.log_probs = log_probs
        self.rewards = rewards
        self.response_masks = response_masks


def compute_policy_log_probs(model, tokenizer, device, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute log probabilities for generated responses."""
        N = len(trajectories)
        policy_log_probs = torch.zeros(N, G, MAX_TOKENS, dtype=torch.float, device=device)
        
        for i, traj in enumerate(trajectories):
            prompt_tokens = tokenizer.encode(traj.prompt, allowed_special={EOF_STR})
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            
            for g in range(G):
                response_tokens = traj.responses[g]
                mask = traj.response_masks[g]
                
                for t in range(MAX_TOKENS):
                    if mask[t] == 0:
                        break
                    input_ids = torch.cat(
                        [
                            prompt_tensor,
                            response_tokens[:t].unsqueeze(0)
                        ], dim=1
                    )
                    logits = model(input_ids)
                    next_token_logits = logits[0, -1, :]
                    log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    token_id = response_tokens[t].item()
                    policy_log_probs[i, g, t] = log_probs[token_id]
        
        return policy_log_probs


# ===================== Base classes (no @ray.remote) =====================

class Generator:
    """Base class for text generation using TransformerLM"""

    def __init__(self):
        self.device = get_device()
        self.generator_model = Transformer(
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            rope_theta=THETA,
            device=self.device,
        )
        load_checkpoint(CHECKPOINT_PATH, self.generator_model, None)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    @torch.no_grad()
    def generate_trajectories(self, prompts: List[str]) -> List[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.

        - For each prompt, generate G responses using self.model
        - Calculate log probabilities for generated tokens
        - Return list of Trajectory objects with prompts, responses, log_probs
        """
        trajs: List[Trajectory] = []

        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(prompt, allowed_special={EOF_STR})
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            keyword = prompt.split()[-1]
            
            rollout_responses = []
            rollout_log_probs = []
            rollout_rewards = []
            rollout_masks = []
            
            for _ in range(G):
                input_ids = prompt_tensor.clone()
                response_log_probs = []
                
                for _ in range(MAX_TOKENS):
                    logits = self.generator_model(input_ids)
                    next_token_logits = logits[0, -1, :]

                    log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(torch.exp(log_probs), num_samples=1)
                    response_log_probs.append(log_probs[next_token.item()])
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                    
                    if next_token.item() in EOF_TOKENS:
                        break
                
                response_tokens = input_ids[0, len(prompt_tokens):].cpu().tolist()
                response_text = self.tokenizer.decode(response_tokens)
                reward = 1.0 if keyword in response_text else 0.0

                # if reward == 1.0:
                #     print(f"Generated response includes keyword '{keyword}': {response_text}")
                
                rollout_responses.append(torch.tensor(response_tokens, dtype=torch.long, device=self.device))
                rollout_log_probs.append(torch.stack(response_log_probs))
                rollout_rewards.append(reward)
                rollout_masks.append(torch.ones(len(response_log_probs), dtype=torch.float, device=self.device))
            
            padded_responses = torch.zeros(G, MAX_TOKENS, dtype=torch.long, device=self.device)
            padded_log_probs = torch.zeros(G, MAX_TOKENS, dtype=torch.float, device=self.device)
            padded_masks = torch.zeros(G, MAX_TOKENS, dtype=torch.float, device=self.device)
            
            for i in range(G):
                seq_len = len(rollout_responses[i])
                padded_responses[i, :seq_len] = rollout_responses[i]
                padded_log_probs[i, :seq_len] = rollout_log_probs[i]
                padded_masks[i, :seq_len] = rollout_masks[i]
            
            traj = Trajectory(
                prompt=prompt,
                responses=padded_responses,
                log_probs=padded_log_probs,
                rewards=torch.tensor(rollout_rewards, dtype=torch.float, device=self.device),
                response_masks=padded_masks,
            )
            trajs.append(traj)

        return trajs


class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self):
        self.device = get_device()
        self.learner_model = Transformer(
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            rope_theta=THETA,
            device=self.device,
        )
        load_checkpoint(CHECKPOINT_PATH, self.learner_model, None)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.optimizer = torch.optim.AdamW(self.learner_model.parameters(), lr=5e-4)
    
    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        rewards = torch.stack([traj.rewards for traj in trajectories])
        rewards_mean = rewards.mean(dim=1, keepdim=True)
        if USE_STD_NORMALIZATION:
            rewards_std = rewards.std(dim=1, keepdim=True)
            return (rewards - rewards_mean) / (rewards_std + ADVANTAGE_EPS)
        else:
            return rewards - rewards_mean
    
    def update_policy(
        self,
        trajectories: List[Trajectory],
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_masks: torch.Tensor,
        steps_per_rollout_batch: int,
        monitor_kl_div: bool=False,
        ref_model: Optional[Transformer]=None,
    ) -> float:
        self.optimizer.zero_grad()
        policy_log_probs = compute_policy_log_probs(
            self.learner_model,
            self.tokenizer,
            self.device,
            trajectories,
        )
        
        loss, _ = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_masks,
            gradient_accumulation_steps=steps_per_rollout_batch,
            loss_type=LOSS_TYPE,
            advantages=advantages,
            old_log_probs=old_log_probs,
            cliprange=0.2,
        )

        gradient_clipping(list(self.learner_model.parameters()), GRAD_NORM_CLIP)
        self.optimizer.step()

        if monitor_kl_div:
            with torch.no_grad():
                ref_log_probs = compute_policy_log_probs(
                    ref_model,
                    self.tokenizer,
                    self.device,
                    trajectories,
                )

                updated_policy_log_probs = compute_policy_log_probs(
                    self.learner_model,
                    self.tokenizer,
                    self.device,
                    trajectories,
                )

                mask = response_masks

                kl_ref_per_sample = ((updated_policy_log_probs - ref_log_probs) * mask).sum(dim=-1)
                kl_divs_ref = kl_ref_per_sample.mean()

                kl_old_per_sample = ((updated_policy_log_probs - policy_log_probs) * mask).sum(dim=-1)
                kl_divs_old = kl_old_per_sample.mean()

                print(f"KL Divergence to reference model: {kl_divs_ref.item():.6f}")
                print(f"KL Divergence to old policy: {kl_divs_old.item():.6f}")

        return loss.item()


# ===================== Combined Actor =====================

@ray.remote(num_gpus=1)
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""
    def __init__(self, monitor_kl_div: bool=False, steps_per_rollout_batch: int=1):
        if torch.cuda.is_available():
            torch.set_default_device("cuda")

        self.device = get_device()
        self.monitor_kl_div = monitor_kl_div
        
        Generator.__init__(self)
        Learner.__init__(self)
        if monitor_kl_div:
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
        
        self.step_count = 0
        self.steps_per_rollout_batch = steps_per_rollout_batch
        self.sync_models()

    def sync_models(self):
        """Sync policy model parameters to generator model."""
        torch.cuda.synchronize()
        self.generator_model.load_state_dict(self.learner_model.state_dict())
    
    def training_step(self, prompts: List[str], monitor_kl_div: bool=False):
        """Perform one complete training step: generate rollout + update policy."""
        generation_start_event = torch.cuda.Event(enable_timing=True)
        generation_end_event = torch.cuda.Event(enable_timing=True)
        learning_start_event = torch.cuda.Event(enable_timing=True)
        learning_end_event = torch.cuda.Event(enable_timing=True)
        weight_sync_start_event = torch.cuda.Event(enable_timing=True)
        weight_sync_end_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.nvtx.range("Generating Step"):
            generation_start_event.record()
            trajectories = self.generate_trajectories(prompts)
            generation_end_event.record()
            advantages = self.compute_advantages(trajectories).unsqueeze(-1)
        batch_old_log_probs = torch.stack(
            [traj.log_probs for traj in trajectories]
        )
        batch_response_masks = torch.stack(
            [traj.response_masks for traj in trajectories]
        )
        
        with torch.cuda.nvtx.range("Learning Step"):
            learning_start_event.record()
            loss = -1.0
            for _ in range(self.steps_per_rollout_batch):
                loss = self.update_policy(
                    trajectories,
                    advantages,
                    batch_old_log_probs,
                    batch_response_masks,
                    self.steps_per_rollout_batch,
                    monitor_kl_div=monitor_kl_div,
                    ref_model=self.ref_model if monitor_kl_div else None,
                )
            learning_end_event.record()

        with torch.cuda.nvtx.range("Weight Sync Step"):
            weight_sync_start_event.record()
            self.sync_models()
            weight_sync_end_event.record()
        torch.cuda.synchronize()

        generation_time = generation_start_event.elapsed_time(generation_end_event)
        learning_time = learning_start_event.elapsed_time(learning_end_event)
        weight_sync_time = weight_sync_start_event.elapsed_time(weight_sync_end_event)

        self.step_count += 1
        print(f"Step {self.step_count}: Loss = {loss:.4f}, avg rewards: {torch.mean(torch.stack([traj.rewards.mean() for traj in trajectories])).item():.4f}")
        print(f"  Generation time: {generation_time:.2f} ms")
        print(f"  Learning time: {learning_time:.2f} ms")
        print(f"  Weight sync time: {weight_sync_time:.2f} ms")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.learner_model.parameters()) if hasattr(self, 'model') else 0
        }


# ===================== Training loop =====================

def run_training(
    prompts: List[str],
    num_steps: int,
    num_workers: int = 1,
    monitor_kl_div: bool=False,
    prompts_per_batch: int=4,
    steps_per_rollout_batch: int=1,
):
    """Run colocated GRPO training with text generation."""
    worker = ColocatedWorker.remote(monitor_kl_div=monitor_kl_div, steps_per_rollout_batch=steps_per_rollout_batch)
    for step in range(num_steps):
        start_idx = step * prompts_per_batch * steps_per_rollout_batch
        end_idx = start_idx + prompts_per_batch * steps_per_rollout_batch
        prompts_step = prompts[start_idx:end_idx]
        ray.get(worker.training_step.remote(prompts_step, monitor_kl_div=monitor_kl_div))

def run_once(
    keywords_file: str,
    num_steps: int,
    num_workers: int = 1,
    monitor_kl_div: bool=False,
    prompts_per_batch: int=4,
    steps_per_rollout_batch: int=1,
):
    """Entry point for training."""
    with open(keywords_file, "r") as f:
        keywords = [line.strip() for line in f.readlines()]
    prompts = make_keyword_inclusion_prompts(keywords)
    run_training(prompts, num_steps, num_workers, monitor_kl_div=monitor_kl_div, prompts_per_batch=prompts_per_batch, steps_per_rollout_batch=steps_per_rollout_batch)


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of training steps")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of colocated workers")
    parser.add_argument("--keywords-file", type=str, required=True,
                       help="Path to file containing keywords, one per line")
    parser.add_argument("--monitor-kl-div", action="store_true",
                       help="Whether to monitor KL divergence during training")
    parser.add_argument("--prompts-per-batch", type=int, default=4,
                       help="Number of prompts to process per batch")
    parser.add_argument("--steps-per-rollout-batch", type=int, default=1,
                       help="Number of gradient steps per rollout batch")
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
        _temp_dir="/app/ray_tmp",
        ignore_reinit_error=True,
    )
    
    try:
        run_once(
            keywords_file=args.keywords_file,
            num_steps=args.steps,
            num_workers=args.workers,
            monitor_kl_div=args.monitor_kl_div,
            prompts_per_batch=args.prompts_per_batch,
            steps_per_rollout_batch=args.steps_per_rollout_batch,
        )
    finally:
        ray.shutdown()
