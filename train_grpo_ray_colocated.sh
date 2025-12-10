#!/bin/bash

uv run -m  cse599o_alignment.train_grpo_ray_colocated \
--tmp-dir /homes/iws/puyuan/ray_tmp \
--keywords-file /homes/iws/puyuan/cse599o/assignment3-rl/cse599o_alignment/prompts/keywords.txt \
--pretrained-ckpt-path /homes/iws/puyuan/cse599o/assignment3-rl/checkpoints/ckpt_pretrained.pt \
--steps 64 \
--prompts-per-batch 32 \
--steps-per-rollout-batch 1 \
--monitor-kl-div
