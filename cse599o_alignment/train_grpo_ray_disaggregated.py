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
from ray.experimental.collective import create_collective_group
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
    def __init__(self, ckpt_path: str,):
        super().__init__(ckpt_path)

    def set_weights(self, state_dict: Dict[str, Any]) -> None:
        torch.cuda.synchronize()
        self.generator_model.load_state_dict(state_dict)


@ray.remote(num_gpus=1)
class LearnerWorker(Learner):
    def __init__(self, ckpt_path: str,):
        super().__init__(ckpt_path)

    def get_weights(self) -> Dict[str, Any]:
        torch.cuda.synchronize()
        return self.learner_model.state_dict()
    
    @ray.method(tensor_transport="nccl")
    def get_weights_rdt(self) -> Dict[str, Any]:
        torch.cuda.synchronize()
        return self.learner_model.state_dict()


# ===================== Training loop =====================

def run_training(
    ckpt_path: str,
    keywords_file: str,
    num_steps: int = 10,
    num_workers: int = 2,
    prompts_per_batch: int = 4,
    use_rdt: bool = False,
    steps_per_rollout_batch: int = 1,
    profile: bool = False,
) -> None:
    """
    Run disaggregated GRPO training.
    
    Args:
        ckpt_path: Path to pretrained model checkpoint
        keywords_file: Path to keywords file
        num_steps: Number of training steps
        num_workers: Must be 2 (generator + learner)
        prompts_per_batch: Number of prompts per batch
        steps_per_rollout_batch: Number of policy update steps per rollout batch
    """
    if num_workers != 2:
        raise ValueError("Disaggregated training requires exactly 2 workers (generator + learner)")
    
    if prompts_per_batch <= 0:
        raise ValueError("prompts_per_batch must be positive")

    with open(keywords_file, "r") as f:
        keywords = [line.strip() for line in f.readlines()]

    print("Creating generator and learner workers...")
    generator = GeneratorWorker.remote(ckpt_path=ckpt_path)
    learner = LearnerWorker.remote(ckpt_path=ckpt_path)

    if use_rdt:
        create_collective_group([generator, learner], backend="nccl")

    with open(keywords_file, "r") as f:
        keywords = [line.strip() for line in f.readlines()]
    prompts = make_keyword_inclusion_prompts(keywords[:prompts_per_batch * steps_per_rollout_batch])

    time_start = time.perf_counter()

    # Generate first batch of trajs
    trajectories_ref = generator.generate_trajectories.remote(prompts)
    
    for step_count in range(num_steps - 1):

        for _ in range(steps_per_rollout_batch):
            loss_ref = learner.update_policy.remote(
                trajectories_ref,
                steps_per_rollout_batch,
            )

        if profile:
            ray.get(loss_ref)

        trajectories_ref = generator.generate_trajectories.remote(prompts)

        if profile:
            transfer_start = time.perf_counter()
        
        if use_rdt:
            updated_weights = learner.get_weights_rdt.remote()
        else:
            updated_weights = learner.get_weights.remote()
        
        # guarantee at most one version behind
        ray.get(generator.set_weights.remote(updated_weights))
        if profile:
            transfer_end = time.perf_counter()
            print(f"Weight transfer time at step {step_count + 1}: {(transfer_end - transfer_start)*1000:.4f} ms.", flush=True)
        else:
            print(f"Step {step_count + 1} weights transferred.", flush=True)

    for _ in range(steps_per_rollout_batch):
        loss_ref = learner.update_policy.remote(
            trajectories_ref,
            steps_per_rollout_batch,
        )
    print(f"Step {num_steps} weights transferred.", flush=True)
    ray.get(loss_ref)

    time_end = time.perf_counter()
    print(f"{num_steps} disaggregated training completed in {(time_end - time_start)*1000:.4f} ms.")

def run_once(
    ckpt_path: str,
    keywords_file: str,
    num_steps: int = 10,
    num_workers: int = 2,
    prompts_per_batch: int = 4,
    use_rdt: bool = False,
    steps_per_rollout_batch: int = 1,
    profile: bool = False,
):
    """Entry point for disaggregated training."""
    run_training(
        ckpt_path=ckpt_path,
        keywords_file=keywords_file,
        num_steps=num_steps,
        num_workers=num_workers,
        prompts_per_batch=prompts_per_batch,
        use_rdt=use_rdt,
        steps_per_rollout_batch=steps_per_rollout_batch,
        profile=profile,
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
    parser.add_argument(
        "--steps-per-rollout-batch",
        type=int,
        default=1,
        help="Number of policy update steps per rollout batch",
    )
    parser.add_argument(
        "--use-rdt",
        action="store_true",
        help="Use Ray Distributed Tensor (RDT) for weight transfer",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    args = parser.parse_args()

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
            ckpt_path=args.pretrained_ckpt_path,
            keywords_file=args.keywords_file,
            num_steps=args.steps,
            num_workers=args.workers,
            prompts_per_batch=args.prompts_per_batch,
            use_rdt=args.use_rdt,
            steps_per_rollout_batch=args.steps_per_rollout_batch,
            profile=args.profile,
        )
    finally:
        ray.shutdown()
