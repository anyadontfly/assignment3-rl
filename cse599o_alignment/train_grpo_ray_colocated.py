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
from cse599o_alignment.grpo import grpo_microbatch_train_step


# ===================== Basic setup =====================

G = 4  # group size (number of responses per prompt)
MAX_BATCH_SIZE = 128
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
CONTEXT_LENGTH = 256
NUM_LAYERS = 4
D_MODEL = 512
NUM_HEADS = 16
D_FF = 1344
THETA = 10000
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


def compute_log_probs(model, tokenizer, device, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute log probabilities for generated responses."""
        N = len(trajectories)
        policy_log_probs = torch.zeros(N, G, MAX_TOKENS, dtype=torch.float, device=device)
        
        prompt_token_lists = [tokenizer.encode(traj.prompt, allowed_special={EOF_STR}) for traj in trajectories]
        prompt_lengths = [len(tokens) for tokens in prompt_token_lists]
        max_prompt_len = max(prompt_lengths)
        
        padded_prompts = torch.zeros(N, max_prompt_len, dtype=torch.long, device=device)
        for i, tokens in enumerate(prompt_token_lists):
            padded_prompts[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        
        batch_prompts = padded_prompts.repeat_interleave(G, dim=0)  # (N*G, max_prompt_len)
        batch_prompt_lengths = torch.tensor([prompt_lengths[i // G] for i in range(N * G)], dtype=torch.long, device=device)
        
        batch_responses = torch.zeros(N * G, MAX_TOKENS, dtype=torch.long, device=device)
        batch_masks = torch.zeros(N * G, MAX_TOKENS, dtype=torch.float, device=device)
        
        for i in range(N):
            for g in range(G):
                batch_idx = i * G + g
                batch_responses[batch_idx] = trajectories[i].responses[g]
                batch_masks[batch_idx] = trajectories[i].response_masks[g]
        
        total_sequences = N * G
        num_batches = (total_sequences + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * MAX_BATCH_SIZE
            end_idx = min(start_idx + MAX_BATCH_SIZE, total_sequences)
            batch_size = end_idx - start_idx
            
            batch_input_ids_list = []
            batch_prompt_lens = []
            
            for i in range(start_idx, end_idx):
                prompt_len = batch_prompt_lengths[i].item()
                prompt_part = batch_prompts[i, :prompt_len]
                response_part = batch_responses[i]
                full_input = torch.cat([prompt_part, response_part])
                
                batch_input_ids_list.append(full_input)
                batch_prompt_lens.append(prompt_len)
            
            max_len = max(len(ids) for ids in batch_input_ids_list)
            padded_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            for idx, ids in enumerate(batch_input_ids_list):
                padded_input_ids[idx, :len(ids)] = ids
            
            logits = model(padded_input_ids)  # (batch_size, max_len, vocab_size)
            
            for idx, seq_idx in enumerate(range(start_idx, end_idx)):
                prompt_len = batch_prompt_lens[idx]
                response_logits = logits[idx, prompt_len - 1 : prompt_len + MAX_TOKENS - 1, :]  # (MAX_TOKENS, vocab_size)
                
                scaled_logits = response_logits / SAMPLING_TEMPERATURE
                log_probs_vocab = torch.log_softmax(scaled_logits, dim=-1)  # (MAX_TOKENS, vocab_size)
                
                response_tokens = batch_responses[seq_idx].unsqueeze(-1)  # (MAX_TOKENS, 1)
                log_probs = torch.gather(log_probs_vocab, dim=1, index=response_tokens).squeeze(-1)  # (MAX_TOKENS,)
                
                n_idx = seq_idx // G
                g_idx = seq_idx % G
                policy_log_probs[n_idx, g_idx] = log_probs
        
        return policy_log_probs


# ===================== Base classes (no @ray.remote) =====================

class Generator:
    """Base class for text generation using TransformerLM"""

    def __init__(self, ckpt_path: str, profile: bool=False):
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
        load_checkpoint(ckpt_path, self.generator_model, None)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.profile = profile
        self.generator_stats = {
            "generation_times": [],
        }

    @torch.no_grad()
    def generate_trajectories(self, prompts: List[str], **kwargs) -> List[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.

        - For each prompt, generate G responses using self.model
        - Calculate log probabilities for generated tokens
        - Return list of Trajectory objects with prompts, responses, log_probs
        """
        N = len(prompts)
        keywords = [prompt.split()[-1] for prompt in prompts]
        
        prompt_token_lists = [self.tokenizer.encode(p, allowed_special={EOF_STR}) for p in prompts]
        prompt_lengths = [len(tokens) for tokens in prompt_token_lists]
        max_prompt_len = max(prompt_lengths)
        
        padded_prompts = torch.zeros(N, max_prompt_len, dtype=torch.long, device=self.device)
        for i, tokens in enumerate(prompt_token_lists):
            padded_prompts[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self.device)
        
        batch_prompts = padded_prompts.repeat_interleave(G, dim=0)  # (N*G, max_prompt_len)
        batch_prompt_lengths = [prompt_lengths[i // G] for i in range(N * G)]
        
        total_sequences = N * G
        all_input_ids = []
        all_log_probs = []
        
        # Process in batches if N*G exceeds MAX_BATCH_SIZE
        num_batches = (total_sequences + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * MAX_BATCH_SIZE
            end_idx = min(start_idx + MAX_BATCH_SIZE, total_sequences)
            batch_size = end_idx - start_idx
            
            input_ids = batch_prompts[start_idx:end_idx].clone()  # (batch_size, max_prompt_len)
            batch_log_probs = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            
            for _ in range(MAX_TOKENS):
                logits = self.generator_model(input_ids)  # (batch_size, seq_len, vocab_size)
                next_token_logits = logits[:, -1, :] / SAMPLING_TEMPERATURE  # (batch_size, vocab_size)
                
                log_probs = torch.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
                next_tokens = torch.multinomial(torch.exp(log_probs), num_samples=1).squeeze(1)  # (batch_size,)
                token_log_probs = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)  # (batch_size,)
                batch_log_probs.append(token_log_probs)
                
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)  # (batch_size, seq_len+1)
                
                for eof_token in EOF_TOKENS:
                    finished |= (next_tokens == eof_token)
                
                if finished.all():
                    break
            
            all_input_ids.append(input_ids)
            all_log_probs.append(torch.stack(batch_log_probs, dim=1))  # (batch_size, actual_len)
        
        max_seq_len = max_prompt_len + MAX_TOKENS
        
        padded_input_ids = []
        padded_log_probs = []
        for input_ids, log_probs in zip(all_input_ids, all_log_probs):
            batch_size, seq_len = input_ids.shape
            if seq_len < max_seq_len:
                padding_ids = torch.zeros(batch_size, max_seq_len - seq_len, dtype=torch.long, device=self.device)
                input_ids = torch.cat([input_ids, padding_ids], dim=1)
                padding_probs = torch.zeros(batch_size, max_seq_len - seq_len, dtype=torch.float, device=self.device)
                log_probs = torch.cat([log_probs, padding_probs], dim=1)
            padded_input_ids.append(input_ids)
            padded_log_probs.append(log_probs)
        
        input_ids = torch.cat(padded_input_ids, dim=0)  # (N*G, max_seq_len)
        batch_log_probs = torch.cat(padded_log_probs, dim=0)  # (N*G, max_seq_len)
        actual_len = batch_log_probs.size(1)
        
        if actual_len < MAX_TOKENS:
            padding = torch.zeros(N * G, MAX_TOKENS - actual_len, dtype=torch.float, device=self.device)
            batch_log_probs = torch.cat([batch_log_probs, padding], dim=1)
        
        batch_responses = torch.zeros(N * G, MAX_TOKENS, dtype=torch.long, device=self.device)
        batch_masks = torch.zeros(N * G, MAX_TOKENS, dtype=torch.float, device=self.device)
        batch_response_texts = []
        
        for i in range(N * G):
            prompt_len = batch_prompt_lengths[i]
            response_tokens = input_ids[i, prompt_len:].cpu().tolist()
            response_len = min(len(response_tokens), MAX_TOKENS)
            
            batch_responses[i, :response_len] = torch.tensor(response_tokens[:response_len], dtype=torch.long, device=self.device)
            
            eof_position = response_len
            for pos, token in enumerate(response_tokens[:response_len]):
                if token in EOF_TOKENS:
                    eof_position = pos
                    break
            batch_masks[i, :eof_position] = 1.0
            
            response_text = self.tokenizer.decode(response_tokens[:response_len])
            batch_response_texts.append(response_text)
        
        batch_rewards = []
        for i in range(N * G):
            keyword = keywords[i // G]
            reward = 1.0 if keyword.lower() in batch_response_texts[i].lower() else 0.0
            batch_rewards.append(reward)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float, device=self.device)
        
        trajs = []
        for i in range(N):
            start_idx = i * G
            end_idx = start_idx + G
            
            traj = Trajectory(
                prompt=prompts[i],
                responses=batch_responses[start_idx:end_idx],  # (G, MAX_TOKENS)
                log_probs=batch_log_probs[start_idx:end_idx],  # (G, MAX_TOKENS)
                rewards=batch_rewards[start_idx:end_idx],  # (G,)
                response_masks=batch_masks[start_idx:end_idx],  # (G, MAX_TOKENS)
            )
            trajs.append(traj)
        
        return trajs
    
    def get_and_clear_generator_stats(self) -> Dict[str, Any]:
        return {}


class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self, ckpt_path: str, profile: bool=False):
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
        load_checkpoint(ckpt_path, self.learner_model, None)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.optimizer = torch.optim.AdamW(self.learner_model.parameters(), 5e-4)
        self.learner_stats = {
            "computed_advantages_times": [],
            "policy_update_times": [],
        }
        self.profile = profile
    
    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        rewards = torch.stack([traj.rewards for traj in trajectories])
        rewards_mean = rewards.mean(dim=1, keepdim=True)
        if USE_STD_NORMALIZATION:
            rewards_std = rewards.std(dim=1, keepdim=True)
            out = (rewards - rewards_mean) / (rewards_std + ADVANTAGE_EPS)
        else:
            out = rewards - rewards_mean
        return out.unsqueeze(-1)
    
    def update_policy(
        self,
        step_index: int,
        trajectories: List[Trajectory],
        steps_per_rollout_batch: int,
        monitor_kl_div: bool=False,
        ref_model: Optional[Transformer]=None,
    ) -> float:
        self.optimizer.zero_grad()

        div = len(trajectories) // steps_per_rollout_batch
        rollout_step_trajectories = trajectories[step_index * div:(step_index + 1) * div]

        advantages = self.compute_advantages(rollout_step_trajectories)

        old_log_probs = torch.stack(
            [traj.log_probs for traj in rollout_step_trajectories]
        )
        response_masks = torch.stack(
            [traj.response_masks for traj in rollout_step_trajectories]
        )

        policy_log_probs = compute_log_probs(
            self.learner_model,
            self.tokenizer,
            self.device,
            rollout_step_trajectories,
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
                ref_log_probs = compute_log_probs(
                    ref_model,
                    self.tokenizer,
                    self.device,
                    rollout_step_trajectories,
                )

                updated_policy_log_probs = compute_log_probs(
                    self.learner_model,
                    self.tokenizer,
                    self.device,
                    rollout_step_trajectories,
                )

                mask = response_masks

                kl_ref_per_sample = ((updated_policy_log_probs - ref_log_probs) * mask).sum(dim=-1)
                kl_divs_ref = kl_ref_per_sample.mean()

                kl_old_per_sample = ((updated_policy_log_probs - policy_log_probs) * mask).sum(dim=-1)
                kl_divs_old = kl_old_per_sample.mean()

                print(f"KL Divergence to reference model: {kl_divs_ref.item():.6f}")
                print(f"KL Divergence to old policy: {kl_divs_old.item():.6f}")

        return loss.item()
    
    def get_and_clear_learner_stats(self) -> Dict[str, Any]:
        stats = self.learner_stats.copy()
        for key in self.learner_stats:
            self.learner_stats[key] = []
        return stats


# ===================== Combined Actor =====================

@ray.remote(num_gpus=1)
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""
    def __init__(
        self,
        ckpt_path: str,
        monitor_kl_div: bool=False,
        steps_per_rollout_batch: int=1,
        profile: bool=False,
    ):
        if torch.cuda.is_available():
            torch.set_default_device("cuda")

        self.device = get_device()
        self.monitor_kl_div = monitor_kl_div
        
        Generator.__init__(self, ckpt_path, profile=profile)
        Learner.__init__(self, ckpt_path, profile=profile)
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
            load_checkpoint(ckpt_path, self.ref_model, None)
        
        self.step_count = 0
        self.steps_per_rollout_batch = steps_per_rollout_batch
        self.profile = profile
        self.sync_models()

    def sync_models(self):
        """Sync policy model parameters to generator model."""
        torch.cuda.synchronize()
        self.generator_model.load_state_dict(self.learner_model.state_dict())
    
    def training_step(self, prompts: List[str], monitor_kl_div: bool=False):
        """Perform one complete training step: generate rollout + update policy."""
        step_start_time = time.perf_counter()

        trajectories = self.generate_trajectories(prompts)
        gen_traj_end_time = time.perf_counter()
        
        loss = -1.0
        for i in range(self.steps_per_rollout_batch):
            loss = self.update_policy(
                i,
                trajectories,
                self.steps_per_rollout_batch,
                monitor_kl_div=monitor_kl_div,
                ref_model=self.ref_model if monitor_kl_div else None,
            )
        policy_update_end_time = time.perf_counter()

        self.sync_models()
        torch.cuda.synchronize()
        step_end_time = time.perf_counter()

        wall_time = (step_end_time - step_start_time) * 1000
        gen_traj_time = (gen_traj_end_time - step_start_time) * 1000
        policy_update_time = (policy_update_end_time - gen_traj_end_time) * 1000
        sync_weights_time = (step_end_time - policy_update_end_time) * 1000

        self.step_count += 1
        avg_reward = torch.stack([traj.rewards for traj in trajectories]).mean().item()
        print(f"Step {self.step_count}: \n"
              f"wall time = {wall_time:.4f} ms, \n"
              f"gen traj time = {gen_traj_time:.4f} ms, \n"
              f"policy update time = {policy_update_time:.4f} ms, \n"
              f"sync weights time = {sync_weights_time:.4f} ms. \n"
              f"Loss = {loss:.4f}, avg rewards: {avg_reward:.4f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        if self.profile:
            learner_stats = self.get_and_clear_learner_stats()
            generator_stats = self.get_and_clear_generator_stats()
            stats = {**learner_stats, **generator_stats}
            return stats
        else:
            return {}


# ===================== Training loop =====================

def run_training(
    ckpt_path: str,
    prompts: List[str],
    num_steps: int,
    num_workers: int = 1,
    monitor_kl_div: bool=False,
    prompts_per_batch: int=4,
    steps_per_rollout_batch: int=1,
    profile: bool=False,
):
    """Run colocated GRPO training with text generation."""
    worker = ColocatedWorker.remote(
        ckpt_path=ckpt_path,
        monitor_kl_div=monitor_kl_div,
        steps_per_rollout_batch=steps_per_rollout_batch,
        profile=profile,
    )
    for step in range(0, num_steps, steps_per_rollout_batch):
        # start_idx = step * prompts_per_batch * steps_per_rollout_batch
        # end_idx = start_idx + prompts_per_batch * steps_per_rollout_batch
        prompts_step = prompts[0:prompts_per_batch * steps_per_rollout_batch]
        ray.get(worker.training_step.remote(prompts_step, monitor_kl_div=monitor_kl_div))

def run_once(
    ckpt_path: str,
    keywords_file: str,
    num_steps: int,
    num_workers: int = 1,
    monitor_kl_div: bool=False,
    prompts_per_batch: int=4,
    steps_per_rollout_batch: int=1,
    profile: bool=False,
):
    """Entry point for training."""
    with open(keywords_file, "r") as f:
        keywords = [line.strip() for line in f.readlines()]
    prompts = make_keyword_inclusion_prompts(keywords)
    run_training(
        ckpt_path,
        prompts,
        num_steps,
        num_workers,
        monitor_kl_div=monitor_kl_div,
        prompts_per_batch=prompts_per_batch,
        steps_per_rollout_batch=steps_per_rollout_batch,
        profile=profile,
    )


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
    parser.add_argument("--tmp-dir", type=str, required=True, 
                        help="Temporary directory for Ray")
    parser.add_argument("--pretrained-ckpt-path", type=str, required=True,
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--profile", action="store_true",
                       help="Whether to enable profiling")
    args = parser.parse_args()
    
    ray.init(
        runtime_env={
            "excludes": [
                ".git/**",                           # git metadata and objects
                ".venv/**",                          # virtual environment
                "submission_*/**",                   # submission folders (6.9GB)
                "checkpoint/**",                     # checkpoint folder (731MB)
                "tests/fixtures/**",                 # test fixtures (large model files)
                "wandb/**",                          # wandb logs
                "*.nsys-rep",                        # profiling files
                # "*.pt", "*.pth", "*.safetensors",   # model weight files
                "*.tar", "*.zip", "*.gz",           # archives
                "__pycache__/**",                   # Python cache
                "*.egg-info/**"                     # package info
            ]
        },
        _temp_dir=args.tmp_dir,
        ignore_reinit_error=True,
    )
    
    try:
        run_once(
            ckpt_path=args.pretrained_ckpt_path,
            keywords_file=args.keywords_file,
            num_steps=args.steps,
            num_workers=args.workers,
            monitor_kl_div=args.monitor_kl_div,
            prompts_per_batch=args.prompts_per_batch,
            steps_per_rollout_batch=args.steps_per_rollout_batch,
            profile=args.profile,
        )
    finally:
        ray.shutdown()
