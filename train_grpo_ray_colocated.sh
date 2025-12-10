#!/bin/bash

uv run -m  cse599o_alignment.train_grpo_ray_colocated \
--tmp-dir /app/ray_tmp \
--keywords-file /app/assignment3-rl/cse599o_alignment/prompts/keywords.txt \
--steps 64 \
--prompts-per-batch 4 \
--steps-per-rollout-batch 1
# --monitor-kl-div