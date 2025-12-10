#!/bin/bash

uv run -m  cse599o_alignment.transfer_bench \
--tmp-dir /app/ray_tmp \
--pretrained-ckpt-path /app/assignment3-rl/checkpoints/ckpt_pretrained.pt \
--num-warmup 3 \
--num-iters 10