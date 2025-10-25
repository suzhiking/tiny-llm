import argparse
from pathlib import Path

import numpy as np

from cs336_basics.preprocessing.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", type=str, default="tinystories")
parser.add_argument("--corpus", type=str, default="TinyStoriesV2-GPT4-valid.txt")

args = parser.parse_args()

data_root_path = Path("~/Courses/CS336/assignment1-basics/data/").expanduser()
corpus = Path(data_root_path / "corpus" / args.corpus)
tokenizer_data_file = Path(data_root_path / "tokenizer_data" / args.tokenizer)

if __name__ == "__main__":
    tokenizer: Tokenizer = Tokenizer.from_files(
        tokenizer_data_file / "vocab.json", tokenizer_data_file / "merges.txt", ["<|endoftext|>"]
    )

    with open(corpus) as f:
        text = f.read()
        tokens = tokenizer.encode(text)

        print("Saving encoded data...")
        np.save(data_root_path / "encoded_data" / "encoded_tinystories_python.npy", tokens)
