from collections import Counter
from itertools import pairwise

from .pretokenization import pretokenize


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


# TODO: Implement the key parts using Rust with PyO3
def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert 256 + len(special_tokens) <= vocab_size
    next_idx = 256
    vocab = {i: bytes([i]) for i in range(256)}
    # vocab = {i: bytes(chr(i), encoding="utf-8") for i in range(256)}
    for tok in special_tokens:
        vocab[next_idx] = tok.encode()
        next_idx += 1
        # print(vocab)
    merges: list[tuple[bytes, bytes]] = list()

    # Get pretokenized word count
    pretoken_cnt = pretokenize(input_path, special_tokens)

    tokens_cnt = {tuple(bytes([b]) for b in s.encode("utf-8")): cnt for s, cnt in pretoken_cnt.items()}

    # b = False
    while next_idx < vocab_size:
        pair_to_merge = None
        max_cnt = 0

        # print(tokens_cnt)
        pair_cnt: Counter[tuple[bytes, bytes]] = Counter()
        for tokens, cnt in tokens_cnt.items():
            for ft, st in pairwise(tokens):
                pair_cnt[(ft, st)] += cnt
        # print(pair_cnt)

        for pair, cnt in pair_cnt.items():
            if cnt > max_cnt or (cnt == max_cnt and pair >= pair_to_merge):
                pair_to_merge = pair
                max_cnt = cnt

        assert pair_to_merge != None

        # if b == False:
        #     print(f"Before merge: {pair_to_merge[0]} and {pair_to_merge[1]}")
        vocab[next_idx] = pair_to_merge[0] + pair_to_merge[1]
        # if b == False:
        #     print(f"After merge: {vocab[next_idx]}")
        #     b = True
        next_idx += 1
        merges.append(pair_to_merge)

        tokens_cnt = {merge_pair(tokens, pair_to_merge): cnt for tokens, cnt in tokens_cnt.items()}

    return vocab, merges


class BPETokenizer:
    def __init__(self) -> None:
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.dict: dict[tuple[bytes], int] = dict()
