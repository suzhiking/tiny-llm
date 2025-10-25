import os
import typing

import torch
from jaxtyping import Int
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import Optimizer


def load_data(x: NDArray[int], batch_size: int, context_length: int, device: str = "mps"):
    x_tensor = torch.from_numpy(x)

    start_idx = torch.randint(len(x) - context_length, (batch_size,), device="cpu")
    offsets: Int[Tensor, " 1 context_length"] = torch.arange(context_length, device="cpu").unsqueeze(0)
    seq_idx: Int[Tensor, " batch_size context_length"] = start_idx.unsqueeze(1) + offsets

    # Advanced indexing
    input_seq = x_tensor[seq_idx].to(device=device, dtype=torch.long)
    targets = x_tensor[seq_idx + 1].to(device=device, dtype=torch.long)

    return input_seq, targets


def load_data_with_idx(start_idx: Tensor, x: NDArray[int], batch_size: int, context_length: int, device: str = "mps"):
    x_tensor = torch.from_numpy(x)

    start_idx = start_idx.to(device="cpu", dtype=torch.long)
    offsets: Int[Tensor, " 1 context_length"] = torch.arange(context_length, device="cpu").unsqueeze(0)
    seq_idx: Int[Tensor, " batch_size context_length"] = start_idx.unsqueeze(1) + offsets

    # Advanced indexing
    input_seq = x_tensor[seq_idx].to(device=device, dtype=torch.long)
    targets = x_tensor[seq_idx + 1].to(device=device, dtype=torch.long)

    return input_seq, targets


def save_checkpoint(
    model: nn.Module, optimizer: Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    states = {"model": model_state, "optimizer": optimizer_state, "iteration": iteration}

    torch.save(states, out)


def load_checkpoint(src, model: nn.Module, optimizer: Optimizer):
    states = torch.load(src)
    model_state = states["model"]
    optimizer_state = states["optimizer"]
    iteration = states["iteration"]

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    return iteration
