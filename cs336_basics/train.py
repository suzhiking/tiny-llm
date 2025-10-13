import argparse
from pathlib import Path

import numpy as np
import wandb

from cs336_basics.preprocessing.data import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, default="../data/encoded_data/")

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

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="gavin",
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

file_path = Path(args.file_path)
token_ids = np.memmap(file_path)

token_cnt = 0


pbar = tqdm(total=args.token_budget)
while token_cnt < args.token_budget:
    batch = load_data(corpus, batch_size=32)
