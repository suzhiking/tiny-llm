import multiprocessing
import os
from collections import Counter
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file: BinaryIO, special_tokens: list[bytes]) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_pretokens(chunks: list[str]):
    result: Counter[str] = Counter()
    for chunk in chunks:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        for match in re.finditer(PAT, chunk):
            result[match.group()] += 1

    return result


def pretokenize(input_path, special_tokens) -> dict[str, int]:
    """
    Return count of pre-tokens
    """
    with open(input_path, encoding="utf-8") as f:
        content = f.read()
        num_processes = 4
        chunks = re.split("|".join(map(re.escape, special_tokens)), content)
        # start_end = zip(boundaries[:-1], boundaries[1:])

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(count_pretokens, [chunks[i::num_processes] for i in range(num_processes)])

        return sum(results, Counter())
