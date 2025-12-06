import os
import typing
from typing import Iterable, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

import numpy as np
import numpy.typing as npt


def data_loading(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device_str: str,
) -> Tuple[Tensor, Tensor]:
    assert len(x.shape) == 1, f"Expected x to be a 1D array of token IDs, but got x of shape {x.shape}"

    if len(x) <= context_length:
        raise ValueError(f"Expected x of length {len(x)} to be longer than context_length {context_length}")
    
    if len(x) - context_length < batch_size + 1:
        raise ValueError(
            f"x of length {len(x)} does not have enough tokens "
            f"to form input/target pairs of shape {(batch_size, context_length)}"
        )
    
    start_indices = torch.randint(0, len(x) - context_length, (batch_size,))
    x_torch = torch.from_numpy(x.astype(np.int64, copy=False))
    input_t = torch.stack([
        x_torch[start : start + context_length] for start in start_indices
    ])
    target_t = torch.stack([
        x_torch[start + 1 : start + 1 + context_length] for start in start_indices
    ])
    return input_t.to(device_str), target_t.to(device_str)

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    r"""Dump all state from model, optimizer, and iteration into `out`."""
    # Use state_dict() for both model and optimizer
    # Use torch.save(obj, out) to dump obj into out
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> int:
    """Load a checkpoint from `src` and recover model and optimizer states."""
    # Use torch.load(src) to recover saved state
    # Call load_state_dict on both model and optimizer
    # Return the saved iteration number
    print(f"Loading checkpoint file: {src}")
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']

def cross_entropy_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    assert len(inputs.shape) == 2, f"Expected inputs have shape (batch_size, vocab_size), but got {inputs.shape}"
    assert len(targets.shape) == 1 and targets.shape[0] == inputs.shape[0], (
        f"Expected targets to have shape (batch_size,), matching inputs' batch size {inputs.shape[0]}, but got {targets.shape}"
    )

    inputs_max = torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = inputs_max + torch.log(torch.sum(torch.exp(inputs - inputs_max), dim=-1, keepdim=True))
    target_inputs = inputs[torch.arange(inputs.size(0), device=inputs.device), targets]
    return (log_sum_exp - target_inputs).mean()

def learning_rate_schedule(t, lr_max, lr_min, t_w, t_c) -> float:
    if t < t_w:
        return t / t_w * lr_max
    elif t >= t_w and t <= t_c:
        return lr_min + 0.5 * (1.0 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min

def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> None:
    params = list(params)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    per_param = [g.detach().float().norm(2) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(per_param), ord=2).item()

    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)