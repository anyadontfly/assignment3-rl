#!/bin/bash

uv run -m cse599o_alignment.train_grpo_ray_disaggregated \
--tmp-dir /homes/iws/puyuan/ray_tmp \
--keywords-file /homes/iws/puyuan/cse599o/assignment3-rl/cse599o_alignment/prompts/keywords.txt \
--pretrained-ckpt-path /homes/iws/puyuan/cse599o/assignment3-rl/checkpoints/ckpt_pretrained.pt \
--steps 32 \
--prompts-per-batch 4 \
--steps-per-rollout-batch 1 \
--workers 2 \
--profile
# --use-rdt
