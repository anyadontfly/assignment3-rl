#!/bin/bash

uv run -m cse599o_alignment.train_grpo_ray_disaggregated \
--tmp-dir /app/ray_tmp \
--keywords-file /app/assignment3-rl/cse599o_alignment/prompts/keywords.txt \
--pretrained-ckpt-path /app/assignment3-rl/checkpoints/ckpt_pretrained.pt \
--steps 5 \
--prompts-per-batch 32 \
--steps-per-rollout-batch 1 \
--workers 2 \
--profile
# --use-rdt
