import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch import Tensor, inference_mode

import wandb
from src.model import Transformer
from src.optimizer import AdamW, cross_entropy
from src.preprocessing.data import load_checkpoint, load_data, load_data_with_idx, save_checkpoint

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors",
    category=UserWarning,
)
# then:

parser = argparse.ArgumentParser()
parser.add_argument("--train_file_path", type=str, default="~/Courses/CS336/data/encoded_data/encoded_tinystories_merged.npy")
parser.add_argument("--valid_file_path", type=str, default="~/Courses/CS336/data/encoded_data/validation/encoded_tinystories_merged.npy")
parser.add_argument("--load_path", type=str)
parser.add_argument("--checkpoint_path", type=str, default="checkpoint/")

# Model hyperparameters
parser.add_argument("--vocab_size", type=int, default=10000, help="Determines the size of token universe")
parser.add_argument("--context_length", type=int, default=256, help="Determine the length of token sequence per sample")
parser.add_argument("--d_model", type=int, default=512, help="Total dimension of embedding vector")
# Usually d_ff = 8/3 d_model
parser.add_argument("--d_ff", type=int, default=1344, help="Hidden dimension for fead-forward layers")
parser.add_argument("--rope_theta", type=float, default=10000, help="Parameter for RoPE")
parser.add_argument("--num_layers", type=int, default=4, help="Depth of transformer blocks")
parser.add_argument(
    "--num_heads", type=int, default=16, help="Number of heads per attention block, must divides d_model"
)
parser.add_argument("--token_budget", type=int, default=40000000, help="How many tokens the model will consume")
parser.add_argument("--batch_size", type=int, default=32)


args = parser.parse_args()

token_budget = args.batch_size * 5000 * args.context_length
target_loss = 2.00

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="z2552wan-university-of-waterloo",
    # Set the wandb project where this run will be logged.
    project="llm",
    # Track hyperparameters and run metadata.
    config={
        "dataset": "TinyStories",
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "token_budget": token_budget,
    },
)

train_file_path = Path(args.train_file_path)
valid_file_path = Path(args.valid_file_path)
checkpoint_path = Path(args.checkpoint_path)
train_token_ids = np.load(train_file_path, mmap_mode="r")

token_cnt = 0

batch_size = args.batch_size
valid_batch_size = batch_size * 10
context_length = args.context_length

model = Transformer(
    d_model=args.d_model,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    theta=args.rope_theta,
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    num_layers=args.num_layers,
    device="mps",
)
model.to("mps")
optimizer = AdamW(model.parameters())

if args.load_path:
    model = load_checkpoint(args.load_path, model, optimizer)


pbar = tqdm.tqdm(total=args.token_budget)
iteration = 1
while token_cnt < args.token_budget:
    batch = load_data(train_token_ids, batch_size=args.batch_size, context_length=args.context_length)
    input: Tensor = batch[0]
    targets: Tensor = batch[1]

    token_cnt += input.numel()

    logits = model(input)
    loss = cross_entropy(logits, targets)

    current_lr = optimizer.param_groups[0]["lr"]
    run.log({"training_loss": loss, "step": iteration, "learning_rate": current_lr})

    loss.backward()
    optimizer.step()

    optimizer.zero_grad()

    pbar.update(input.numel())

    if iteration % 10== 0:
        model.eval()
        with inference_mode():
            valid_token_ids = np.load(valid_file_path, mmap_mode="r")
            valid_loss = 0.0
            total_tokens = 0
            max_start = len(valid_token_ids) - valid_batch_size * context_length
            step = valid_batch_size * context_length
            for i in tqdm.tqdm(range(0, max_start, step), desc=f"valid[{iteration}]", dynamic_ncols=True):
                start_idx = torch.arange(valid_batch_size, dtype=torch.long) * context_length + i
                batch = load_data_with_idx(start_idx, valid_token_ids, valid_batch_size, context_length)
                input: Tensor = batch[0]
                targets: Tensor = batch[1]
                logits = model(input)
                valid_loss += cross_entropy(logits, targets, "sum").item()
                total_tokens += targets.numel()

            mean_loss = valid_loss / total_tokens
            run.log({"valid_loss": mean_loss})

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d")
        save_checkpoint(model, optimizer, iteration, checkpoint_path / (current_datetime + f"-{iteration}"))

        if mean_loss <= target_loss:
            break

        model.train()

    iteration += 1

run.finish()
