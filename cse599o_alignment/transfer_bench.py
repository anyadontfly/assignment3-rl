import time
import argparse
import ray
from ray.experimental.collective import create_collective_group

from cse599o_alignment.train_grpo_ray_disaggregated import (
    GeneratorWorker,
    LearnerWorker,
)


def transfer_once(generator, learner, use_rdt: bool):
    """
    Transfer weights once from learner to generator.
    """
    time_start = time.perf_counter()
    if use_rdt:
        updated_weights = learner.get_weights_rdt.remote()
    else:
        updated_weights = learner.get_weights.remote()
    ray.get(generator.set_weights.remote(updated_weights))
    time_end = time.perf_counter()
    return (time_end - time_start) * 1000

def main(ckpt_path: str, num_warmup, num_bench):
    generator, learner = GeneratorWorker.remote(ckpt_path), LearnerWorker.remote(ckpt_path)
    create_collective_group([generator, learner], backend="nccl")

    no_rdt_times = []
    for _ in range(num_warmup):
        transfer_once(generator, learner, use_rdt=False)
    for _ in range(num_bench):
        no_rdt_times.append(transfer_once(generator, learner, use_rdt=False))

    rdt_times = []
    for _ in range(num_warmup):
        transfer_once(generator, learner, use_rdt=True)
    for _ in range(num_bench):
        rdt_times.append(transfer_once(generator, learner, use_rdt=True))

    print(f"Weight transfer without RDT: avg = {sum(no_rdt_times)/len(no_rdt_times):.4f} ms, "
          f"min = {min(no_rdt_times):.4f} ms, "
          f"max = {max(no_rdt_times):.4f} ms.")
    print(f"Weight transfer with RDT: avg = {sum(rdt_times)/len(rdt_times):.4f} ms, "
          f"min = {min(rdt_times):.4f} ms, "
          f"max = {max(rdt_times):.4f} ms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaggregated GRPO Training")
    parser.add_argument(
        "--pretrained-ckpt-path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        required=True,
        help="Temporary directory for Ray",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of benchmark iterations",
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
        main(args.pretrained_ckpt_path, args.num_warmup, args.num_iters)
    finally:
        ray.shutdown()