import argparse
import heapq
import json
import logging
import multiprocessing
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from typing import BinaryIO

import regex as re
from sortedcontainers import SortedSet
from tqdm import tqdm

from cs336_basics.commons import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def count_pretokens(args):
    result: Counter[str] = Counter()
    start_end: list[int, int] = args[0]
    input_path: str = args[1]
    special_tokens: list[str] = args[2]

    with open(input_path, "rb") as f:
        for start, end in start_end:
            f.seek(start)
            content = f.read(end - start).decode("utf-8", errors="ignore")

            chunks = re.split("|".join(map(re.escape, special_tokens)), content)
            for chunk in chunks:
                for match in re.finditer(PAT, chunk):
                    result[match.group()] += 1

    return result


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
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


def pretokenize(input_path: str, special_tokens: list[str]) -> dict[str, int]:
    """
    Return count of pre-tokens
    """
    with open(input_path, "rb") as f:
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        # If the file size is small, don't use parallelism
        num_processes = 10
        boundaries = find_chunk_boundaries(f, num_processes * 10000, b"<|endoftext|>")
        # chunks = re.split("|".join(map(re.escape, special_tokens)), content)
        start_end = list(zip(boundaries[:-1], boundaries[1:]))

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(
            count_pretokens,
            [(start_end[i::num_processes], input_path, special_tokens) for i in range(num_processes)],
        )

    return sum(results, Counter())


def str_to_bytes_tuple(s: str):
    return tuple(bytes([b]) for b in s.encode("utf-8"))



def bytes_to_unicode_str(b: bytes, encoder: dict[int, str]):
    return "".join(encoder[num] for num in tuple(b))


def merge_pair(tokens: tuple[bytes], pair: tuple[bytes, bytes]):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    return tuple(new_tokens)


_SPECIAL = set()
_MERGES = []


class DescBytes:
    __slots__ = ("b",)

    def __init__(self, b: bytes):
        self.b = b

    def __lt__(self, other: "DescBytes"):
        return self.b > other.b

    def __eq__(self, other: object):
        return isinstance(other, DescBytes) and self.b == other.b

    def __repr__(self):
        return f"DescBytes({self.b!r})"


def heap_key(count: int, pair: tuple[bytes, bytes]):
    b1, b2 = pair
    return (-count, (DescBytes(b1), DescBytes(b2)))


def unheap_key(key):
    negc, (d1, d2) = key
    return (-negc, (d1.b, d2.b))


def _merge(token: tuple[bytes], merges):
    for pair in merges:
        token = merge_pair(token, pair)

    return token


def merge_batch(tokens: list[tuple[bytes]]):
    res = []
    for token in tqdm(tokens):
        res.append(_merge(token))

    return res


# TODO: Implement the key parts using Rust with PyO3
def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], output_path: str | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert 256 + len(special_tokens) <= vocab_size
    next_idx = 256
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[next_idx] = tok.encode()
        next_idx += 1
    merges: list[tuple[bytes, bytes]] = list()

    # Get pretokenized word count
    pretoken_cnt = pretokenize(input_path, special_tokens)

    tokens_cnt = {tuple(bytes([b]) for b in s.encode("utf-8")): cnt for s, cnt in pretoken_cnt.items()}
    pair_cnt = Counter()
    idx_to_pair: dict[int, tuple[bytes, bytes]] = dict()
    idx_cnt: dict[int, int] = dict()
    pair_to_idxs: dict[tuple[bytes, bytes], set[int]] = defaultdict(SortedSet)
    neighbors: dict[int, list[int | None, int | None]] = defaultdict(lambda: [None, None])
    idx = 1
    for tokens, cnt in tokens_cnt.items():
        prev_idx = None
        for ft, st in pairwise(tokens):
            pair_cnt[(ft, st)] += cnt

            idx_to_pair[idx] = (ft, st)
            idx_cnt[idx] = cnt  # This won't change
            pair_to_idxs[(ft, st)].add(idx)

            if prev_idx is not None:
                neighbors[prev_idx][1] = idx
                neighbors[idx][0] = prev_idx

            prev_idx = idx
            idx += 1

    heap: list[tuple[int, tuple[bytes, bytes]]] = [heap_key(cnt, pair) for pair, cnt in pair_cnt.items()]
    heapq.heapify(heap)

    pbar = tqdm(total=vocab_size - next_idx)
    while next_idx < vocab_size:
        while heap:
            cnt, pair_to_merge = unheap_key(heapq.heappop(heap))
            if cnt == pair_cnt[pair_to_merge]:
                break

        # if next_idx == 262:
        #     print(cnt, pair_to_merge)
        # if next_idx == 263:
        #     print(cnt, pair_to_merge)
        merged_token: bytes = pair_to_merge[0] + pair_to_merge[1]
        s = sorted(pair_to_idxs[pair_to_merge])
        for idx in s:
            
            if idx_to_pair[idx] != pair_to_merge:
                continue

            left_idx = neighbors[idx][0]
            right_idx = neighbors[idx][1]

            if left_idx and right_idx:
                # pair is in the middle of word
                neighbors[left_idx][1] = right_idx
                neighbors[right_idx][0] = left_idx

                left_pair = idx_to_pair[left_idx]
                right_pair = idx_to_pair[right_idx]
                pair_cnt[left_pair] -= idx_cnt[left_idx]
                pair_cnt[right_pair] -= idx_cnt[right_idx]
                if pair_cnt[left_pair] > 0:
                    heapq.heappush(heap, heap_key(pair_cnt[left_pair], left_pair))
                if pair_cnt[right_pair] > 0:
                    heapq.heappush(heap, heap_key(pair_cnt[right_pair], right_pair))

                idx_to_pair[left_idx] = (left_pair[0], merged_token)
                idx_to_pair[right_idx] = (merged_token, right_pair[1])
                new_left_pair = idx_to_pair[left_idx]
                new_right_pair = idx_to_pair[right_idx]
                pair_cnt[new_left_pair] += idx_cnt[left_idx]
                pair_cnt[new_right_pair] += idx_cnt[right_idx]
                heapq.heappush(heap, heap_key(pair_cnt[new_left_pair], new_left_pair))
                heapq.heappush(heap, heap_key(pair_cnt[new_right_pair], new_right_pair))

                pair_to_idxs[left_pair].discard(left_idx)
                pair_to_idxs[right_pair].discard(right_idx)
                pair_to_idxs[new_left_pair].add(left_idx)
                pair_to_idxs[new_right_pair].add(right_idx)
            elif left_idx:
                neighbors[left_idx][1] = None
                left_pair = idx_to_pair[left_idx]
                pair_cnt[left_pair] -= idx_cnt[left_idx]
                if pair_cnt[left_pair] > 0:
                    heapq.heappush(heap, heap_key(pair_cnt[left_pair], left_pair))
                idx_to_pair[left_idx] = (left_pair[0], merged_token)
                new_left_pair = idx_to_pair[left_idx]
                pair_cnt[new_left_pair] += idx_cnt[left_idx]
                heapq.heappush(heap, heap_key(pair_cnt[new_left_pair], new_left_pair))
                pair_to_idxs[left_pair].discard(left_idx)
                pair_to_idxs[new_left_pair].add(left_idx)
            elif right_idx:
                neighbors[right_idx][0] = None

                right_pair = idx_to_pair[right_idx]
                pair_cnt[right_pair] -= idx_cnt[right_idx]
                if pair_cnt[right_pair] > 0:
                    heapq.heappush(heap, heap_key(pair_cnt[right_pair], right_pair))

                idx_to_pair[right_idx] = (merged_token, right_pair[1])
                new_right_pair = idx_to_pair[right_idx]
                pair_cnt[new_right_pair] += idx_cnt[right_idx]
                heapq.heappush(heap, heap_key(pair_cnt[new_right_pair], new_right_pair))

                pair_to_idxs[right_pair].discard(right_idx)
                pair_to_idxs[new_right_pair].add(right_idx)

        del pair_cnt[pair_to_merge]
        del pair_to_idxs[pair_to_merge]

        vocab[next_idx] = merged_token
        next_idx += 1
        merges.append(pair_to_merge)

        pbar.update(1)

    if output_path:
        gpt2_byte_encoder = gpt2_bytes_to_unicode()

        vocab_output_path = output_path + "/vocab.json"
        merges_output_path = output_path + "/merges.txt"
        os.makedirs(output_path, exist_ok=True)

        serializable_vocab = {bytes_to_unicode_str(b, gpt2_byte_encoder): idx for idx, b in vocab.items()}
        serializable_merges = [
            (bytes_to_unicode_str(b1, gpt2_byte_encoder), bytes_to_unicode_str(b2, gpt2_byte_encoder))
            for (b1, b2) in merges
        ]

        with open(vocab_output_path, "w") as json_file:
            json.dump(serializable_vocab, json_file, indent=4)
        with open(merges_output_path, "w") as file:
            for tok1, tok2 in serializable_merges:
                file.write(f"{tok1} {tok2}\n")

    return vocab, merges


# --- Worker state (top-level, so it's picklable) ---
_WORKER_SPECIAL = set()
_WORKER_MERGES = []


def _init_worker(merges, special_tokens):
    global _WORKER_MERGES, _WORKER_SPECIAL
    _WORKER_MERGES = merges
    _WORKER_SPECIAL = set(special_tokens or [])


def _merge_local(token: tuple[bytes]):
    for pair in _WORKER_MERGES:
        token = merge_pair(token, pair)
    return token


def _encode_batch(batch: list[str]) -> list[tuple[bytes]]:
    out = []
    sp = _WORKER_SPECIAL
    for s in batch:
        if sp and s in sp:
            out.append((s.encode("utf-8"),))
        else:
            out.append(_merge_local(str_to_bytes_tuple(s)))
    return out


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.vocab_rev = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as file:
            raw_vocab = json.load(file)
            vocab = {}
            vocab = {
                int(idx): bytes([gpt2_byte_decoder[tok] for tok in vocab_item])
                for vocab_item, idx in tqdm(raw_vocab.items(), "Constructing vocab")
            }
            # pbar = tqdm(total_size=raw_vocab.items())
            # for vocab_item, idx in raw_vocab.items():

        with open(merges_filepath) as file:
            raw_merges = list()
            for line in file.readlines():
                tok1, tok2 = line.strip().split(" ")
                raw_merges.append((tok1, tok2))
            merges = [
                (bytes([gpt2_byte_decoder[tok] for tok in s1]), bytes([gpt2_byte_decoder[tok] for tok in s2]))
                for s1, s2 in tqdm(raw_merges, "Constructing merges")
            ]

        return cls(vocab, merges, special_tokens)

    def merge_pair(self, tokens: tuple[int], pair: tuple[int, int]):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return tuple(new_tokens)

    def str_to_id_tuple(self, s: str):
        return tuple(self.vocab_rev(bytes([b])) for b in s.encode("utf-8"))


    def _merge(self, token: tuple[int]):
        for pair in self.merges:
            token = self.merge_pair(token, pair)

        return token

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            # Sort it to make sure the longer patterns are captured first
            special_pat = "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
            split_pat = f"({special_pat})"
            parts = re.split(split_pat, text)

            pretokens = []
            for part in tqdm(parts, "Constructing pretokens"):
                if part in self.special_tokens:
                    pretokens.append(part)
                else:
                    pretokens.extend(re.findall(PAT, part))
        else:
            pretokens: list[str] = re.findall(PAT, text)

        # ---- Parallel merge (fixed) ----
        # os.environ.setdefault("OMP_NUM_THREADS", "1")
        # os.environ.setdefault("MKL_NUM_THREADS", "1")

        # ctx = multiprocessing.get_context("spawn")               
        # workers = max(1, (os.cpu_count() or 4) - 1)
        # print(f"Number of workers: {workers}")

        # Batch pretokens to reduce IPC/pickling overhead
        # BATCH = 100000  # tune: 20k–100k often good
        # batches = [pretokens[i:i+BATCH] for i in tqdm(range(0, len(pretokens), BATCH), "Batching pretokens for parallel processing")]

        merged_tokens: list[tuple[bytes]] = []
        # with ProcessPoolExecutor(
        #     max_workers=workers,
        #     mp_context=ctx,
        #     initializer=_init_worker,
        #     initargs=(self.merges, self.special_tokens or []),
        # ) as ex:
        #     it = ex.map(_encode_batch, batches, chunksize=1)
        #     for batch_out in tqdm(it, total=len(batches), desc="Merging batches"):
        #         # flatten incrementally (keeps memory bounded)
        #         for tup in batch_out:
        #             merged_tokens.append(tup)

        # print(merged_tokens)
        # # merged_tokens = []
        for pretok in tqdm(pretokens, "Encoding pretokens"):
            if self.special_tokens and pretok in self.special_tokens:
                # # logger.debug((pretok.encode("utf-8"),))
                merged_tokens.append((pretok.encode("utf-8"),))
            else:
                merged_tokens.append(self._merge(str_to_bytes_tuple(pretok)))

        return [self.vocab_rev[tok] for tok in [b for tup in merged_tokens for b in tup]]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return sum([self.encode(s) for s in iterable], start=[])

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[id] for id in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")


if __name__ == "__main__":
    # logger.debug("Start BPE tokenizer training")
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Input path of corpus text")
    parser.add_argument("-o", "--output_path", type=str, help="Output path trained tokenizer data")
    parser.add_argument("--vocab_size", type=int, default=256, help="Maximum vocabulary size")
    parser.add_argument("--special_tokens", type=list[str], default=["<|endoftext|>"], help="Special tokens")

    args = parser.parse_args()

    train_bpe(args.input_path, args.vocab_size, args.special_tokens, args.output_path)
    # logger.debug("Training completed")
