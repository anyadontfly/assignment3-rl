"""
GRPO: Disaggregated Asynchronous Training Loop
------------------------------------------------
Separates Generator and Learner into different GPU workers for asynchronous training.
Generator produces trajectories while Learner updates policy in parallel.
"""

import argparse
import time
import numpy as np
import ray
import torch
from typing import List, Dict, Any, Optional

from cse599o_alignment.train_grpo_ray_colocated import (
    make_keyword_inclusion_prompts,
    Trajectory,
    Generator,
    Learner,
)


# ===================== Actors =====================

@ray.remote(num_gpus=1)
class GeneratorWorker(Generator):
    """Generator Ray actor with dedicated GPU."""

    def __init__(self):
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        super().__init__()

    def generate_trajectories(self, prompts: List[str], transfer_ref=None) -> List[Trajectory]:
        """
        Generate trajectories, optionally waiting for weight transfer first.
        
        Args:
            prompts: List of prompt strings
            transfer_ref: Optional Ray object ref to wait for weight sync
            
        Returns:
            List of Trajectory objects
        """
        if transfer_ref is not None:
            ray.get(transfer_ref)
        
        return super().generate_trajectories(prompts)

    def set_weights(self, weights):
        """Update generator model weights from learner."""
        self.generator_model.load_state_dict(weights)
        torch.cuda.synchronize()
        return True


@ray.remote(num_gpus=1)
class LearnerWorker(Learner):
    """Learner Ray actor with dedicated GPU."""

    def __init__(self):
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        super().__init__()

    def get_weights(self, loss_ref=None):
        """
        Return current policy model weights.
        
        Args:
            loss_ref: Optional Ray object ref to wait for loss computation
            
        Returns:
            Model state dict
        """
        if loss_ref is not None:
            ray.get(loss_ref)
        
        torch.cuda.synchronize()
        return self.learner_model.state_dict()


# ===================== Training loop =====================

def train_disaggregated(
    generator: "ray.actor.ActorHandle",
    learner: "ray.actor.ActorHandle",
    num_steps: int,
    keywords: List[str],
    prompts_per_batch: int,
) -> float:
    """
    Args:
        generator: Generator worker actor handle
        learner: Learner worker actor handle
        num_steps: Number of training steps
        keywords: List of keywords for prompts
        prompts_per_batch: Number of prompts per batch
        
    Returns:
        Final loss value
    """
    all_keywords = keywords[:num_steps * prompts_per_batch]
    prompts_list: List[List[str]] = [
        make_keyword_inclusion_prompts(all_keywords[i:i + prompts_per_batch])
        for i in range(0, len(all_keywords), prompts_per_batch)
    ]

    trajs_ref = generator.generate_trajectories.remote(prompts_list[0])
    transfer_ref = None
    
    for step in range(num_steps - 1):
        loss_ref = learner.update_policy.remote(trajs_ref)
        trajs_ref = generator.generate_trajectories.remote(
            prompts_list[step + 1],
            transfer_ref=transfer_ref,
        )
        
        weights_ref = learner.get_weights.remote(loss_ref=loss_ref)
        transfer_ref = generator.set_weights.remote(weights_ref)
    
    loss_ref = learner.update_policy.remote(trajs_ref)
    final_loss = ray.get(loss_ref)
    
    return final_loss


def run_training(
    keywords_file: str,
    num_steps: int = 10,
    num_workers: int = 2,
    prompts_per_batch: int = 4,
) -> None:
    """
    Run disaggregated GRPO training.
    
    Args:
        keywords_file: Path to keywords file
        num_steps: Number of training steps
        num_workers: Must be 2 (generator + learner)
        prompts_per_batch: Number of prompts per batch
    """
    if num_workers != 2:
        raise ValueError("Disaggregated training requires exactly 2 workers (generator + learner)")
    
    if prompts_per_batch <= 0:
        raise ValueError("prompts_per_batch must be positive")

    with open(keywords_file, "r") as f:
        keywords = [line.strip() for line in f.readlines()]

    print("Creating generator and learner workers...")
    generator = GeneratorWorker.remote()
    learner = LearnerWorker.remote()

    print(f"Starting disaggregated training for {num_steps} steps...")
    start_time = time.perf_counter()
    
    final_loss = train_disaggregated(
        generator=generator,
        learner=learner,
        num_steps=num_steps,
        keywords=keywords,
        prompts_per_batch=prompts_per_batch,
    )
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"\nTraining completed!")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Time per step: {elapsed / num_steps:.2f} seconds")


def run_once(
    keywords_file: str,
    num_steps: int = 10,
    num_workers: int = 2,
    prompts_per_batch: int = 4,
    **kwargs
):
    """Entry point for disaggregated training."""
    run_training(
        keywords_file=keywords_file,
        num_steps=num_steps,
        num_workers=num_workers,
        prompts_per_batch=prompts_per_batch,
    )



# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaggregated GRPO Training")
    parser.add_argument(
        "--keywords-file",
        type=str,
        required=True,
        help="Path to keywords file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of training steps",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of workers (must be 2)",
    )
    parser.add_argument(
        "--prompts-per-batch",
        type=int,
        default=4,
        help="Number of prompts per batch",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        required=True,
        help="Temporary directory for Ray",
    )
    parser.add_argument(
        "--pretrained-ckpt-path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument("--steps-per-rollout-batch", type=int, default=1)
    parser.add_argument("--monitor-kl-div", action="store_true")
    args = parser.parse_args()

    import cse599o_alignment.train_grpo_ray_colocated as colocated_module
    colocated_module.CHECKPOINT_PATH = args.pretrained_ckpt_path

    ray.init(
        runtime_env={
            "excludes": [
                ".git/**",
                ".venv/**",
                "submission_*/**",
                "checkpoint/**",
                "tests/fixtures/**",
                "wandb/**",
                "*.nsys-rep",
                "*.tar",
                "*.zip",
                "*.gz",
                "__pycache__/**",
                "*.egg-info/**",
            ]
        },
        _temp_dir=args.tmp_dir,
        ignore_reinit_error=True,
    )

    try:
        run_once(
            keywords_file=args.keywords_file,
            num_steps=args.steps,
            num_workers=args.workers,
            prompts_per_batch=args.prompts_per_batch,
        )
    finally:
        ray.shutdown()
