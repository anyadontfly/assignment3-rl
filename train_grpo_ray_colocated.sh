#!/bin/bash

uv run -m  cse599o_alignment.train_grpo_ray_colocated \
--keywords-file /app/assignment3-rl/cse599o_alignment/prompts/keywords.txt \
--steps 64 \
--prompts-per-batch 4 \
--steps-per-rollout-batch 1