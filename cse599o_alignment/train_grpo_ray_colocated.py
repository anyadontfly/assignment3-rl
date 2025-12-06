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
import tiktoken
import time
from typing import List, Dict, Any, Optional
import numpy as np

from cse599o_basics.model import Transformer, softmax
from cse599o_basics.optimizer import AdamW
from cse599o_basics.utils import save_checkpoint, load_checkpoint
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    gradient_clipping
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
CHECKPOINT_PATH = "./checkpoints/ckpt_pretrained.pt"
MAX_TOKENS = 512
EOF_STR = "<|endoftext|>"
EOF_TOKENS = tiktoken.get_encoding("gpt2").encode(EOF_STR)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Data container =====================

def keyword_inclusion_reward_fn(response, keyword) -> Dict[str, float]:
    if keyword in response:
        return {"reward": 1.0}
    else:
        return {"reward": 0.0}


# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""

    def __init__(
        self,
        prompts: List[str],  # shape: [G]
        responses: List[str],  # shape: [G]
        rewards: torch.Tensor,  # shape: [G]
        log_probs: torch.Tensor,  # shape: [G]
        values: Optional[torch.Tensor] = None,  # shape: [G]
    ):
        self.prompts = prompts
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.values = values


# ===================== Base classes (no @ray.remote) =====================

class Generator:
    """Base class for text generation using TransformerLM"""

    def __init__(self):
        self.device = get_device()
        # TODO: Initialize the TransformerLM model
        self.model = Transformer(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            theta=THETA,
        )
        load_checkpoint(CHECKPOINT_PATH, self.model, None)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def generate_trajectories(self, prompts: List[str]) -> List[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.

        TODO: Implement this method
        - For each prompt, generate G responses using self.model
        - Calculate log probabilities for generated tokens
        - Return list of Trajectory objects with prompts, responses, log_probs
        """
        self.model.eval()
        trajs: List[Trajectory] = []

        with torch.no_grad():
            for prompt in prompts:
                prompt_tokens = self.tokenizer.encode(prompt)
                prompt_len = len(prompt_tokens)

                group_responses: List[str] = []
                group_log_probs: List[torch.Tensor] = []

                for _ in range(G):
                    tokens = list(prompt_tokens)
                    token_logprobs: List[torch.Tensor] = []

                    while True:
                        input_tokens = tokens[-CONTEXT_LENGTH:]
                        input_ids = torch.tensor(
                            [input_tokens],
                            dtype=torch.long,
                            device=self.device,
                        )
                        logits = self.model(input_ids)[0, -1, :]
                        log_probs_vocab = torch.log_softmax(logits, dim=-1)
                        probs_vocab = torch.exp(log_probs_vocab)
                        next_id = torch.multinomial(probs_vocab, num_samples=1).item()
                        token_logprobs.append(log_probs_vocab[next_id])

                        tokens.append(next_id)
                        if (
                            len(tokens) >= len(EOF_TOKENS)
                            and tokens[-len(EOF_TOKENS):] == EOF_TOKENS
                        ):
                            break
                        if len(tokens) - prompt_len >= MAX_TOKENS:
                            break

                    generated_tokens = tokens[prompt_len:]
                    response_text = self.tokenizer.decode(generated_tokens)
                    group_responses.append(response_text)

                    token_logprobs_tensor = torch.stack(token_logprobs).to(self.device)
                    group_log_probs.append(token_logprobs_tensor)

                traj = Trajectory(
                    prompts=[prompt] * G,
                    responses=group_responses,
                    log_probs=group_log_probs,
                    rewards=None,
                )
                trajs.append(traj)

        return trajs


class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self):
        self.device = get_device()
        # TODO: Initialize the same TransformerLM model as Generator
        self.model = Transformer(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            theta=THETA,
        )
        load_checkpoint(CHECKPOINT_PATH, self.model, None)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
    
    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        # TODO: Implement GRPO advantage computation
        # This should implement the group-relative advantage computation
        # that's central to GRPO algorithm
        rollout_responses: List[str] = []
        for traj in trajectories:
            rollout_responses.extend(traj.responses)
        
        if len(rollout_responses) == 0:
            return torch.empty(0, device=self.device)
        
        keyword = "alignment"
        repeated_ground_truths = [keyword] * len(rollout_responses)
        advantages, raw_rewards, _ = compute_group_normalized_reward(
            reward_fn=keyword_inclusion_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=G,
            advantage_eps=1e-8,
            normalized_by_std=True,
        )
        advantages = advantages.to(self.device)
        raw_rewards = raw_rewards.to(self.device)
        
        for group_idx, traj in enumerate(trajectories):
            start = group_idx * G
            end = start + G
            traj.rewards = raw_rewards[start:end]
        return advantages
    
    def update_policy(self, trajectories: List[Trajectory]) -> float:
        """Perform one policy update step."""
        # TODO: Implement GRPO/PPO policy update
        # 1. Compute advantages
        # 2. Compute policy gradient loss
        # 3. Perform optimizer step
        # 4. Return loss value
        
        loss = torch.tensor(0.0, device=self.device)
        return float(loss.item())


# ===================== Combined Actor =====================

@ray.remote
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""
    def __init__(self):
        Generator.__init__(self)
        Learner.__init__(self)
        self.step_count = 0
    
    def training_step(self, prompts: List[str]) -> Dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""
        # Generate trajectories for the batch of prompts
        trajectories = self.generate_trajectories(prompts)
        
        # Update policy using GRPO
        loss = self.update_policy(trajectories)
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'loss': loss,
            'num_trajectories': len(trajectories),
            'avg_reward': float(torch.cat([traj.rewards for traj in trajectories]).mean()) if trajectories else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self, 'model') else 0
        }


# ===================== Training loop =====================

def run_training(num_steps: int = 10, num_workers: int = 1):
    """Run colocated GRPO training with text generation."""
    
    # Create workers  
    
    # TODO: Define training prompts
    
    for step in range(num_steps):
        # TODO: 
        pass
        
    # Get final statistics
    pass


def run_once(num_steps: int = 10, num_workers: int = 1):
    """Entry point for training."""
    run_training(num_steps, num_workers)


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of training steps")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of colocated workers")
    args = parser.parse_args()
    
    ray.init(ignore_reinit_error=True)
    
    try:
        run_once(num_steps=args.steps, num_workers=args.workers)
    finally:
        ray.shutdown()
