import math
from math import sqrt

import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn, sigmoid


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        variance = 2 / (in_features * out_features)
        std = sqrt(variance)

        self.weight = nn.Parameter(torch.zeros([out_features, in_features], device=self.device, dtype=self.dtype))
        torch.nn.init.trunc_normal_(self.weight, 0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
        return y


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight: Float[Tensor, " vocab_size d_model"] = nn.Parameter(
            torch.zeros([num_embeddings, embedding_dim], device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> torch.Tensor:
        # output = torch.index_select(self.weights, 0, token_ids)
        out = self.weight[token_ids]
        return out


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.kwargs = {"device": device, "dtype": dtype}
        self.gain = nn.Parameter(torch.ones(d_model, **self.kwargs))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x: Tensor = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) / self.d_model + self.eps)
        rms_norm: Tensor = (x / rms) * self.gain
        return rms_norm.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.kwargs = {"device": device, "dtype": dtype}
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)
        self.SiLU = SiLU()

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.W2(self.SiLU(self.W1(x)) * self.W3(x))


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # Precompute sin and cos values
        sin_val, cos_val = self.rope_cache(d_k, max_seq_len, theta)
        self.register_buffer("sin_val", sin_val)
        self.register_buffer("cos_val", cos_val)

    def rope_cache(self, d_k: int, max_seq_len: int, base: int) -> Tensor:
        inv_freq: Tensor = 1 / (base ** (torch.arange(0, d_k // 2) * 2 / d_k))
        pos: Tensor = torch.arange(max_seq_len)
        angles: Tensor = pos[:, None] * inv_freq[None, :]
        return torch.sin(angles), torch.cos(angles)

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Float[Tensor, "... seq_len"]
    ) -> torch.Tensor:
        sin: Float[Tensor, "seq_len half_dim"] = self.sin_val[token_positions]
        cos: Float[Tensor, "seq_len half_dim"] = self.cos_val[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)
        return x_rot


def softmax(x: Tensor, i: int) -> Tensor:
    x_max = torch.max(x, dim=i, keepdim=True).values
    x = x - x_max
    x_exp = torch.exp(x)
    x_sum = torch.sum(x_exp, dim=i, keepdim=True)
    print(x.shape)
    return x_exp / x_sum


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    P = Q @ torch.transpose(K, -1, -2) / math.sqrt(d_k)

    if mask is not None:
        P = P.masked_fill(~mask, torch.finfo(P.dtype).min)

    P = softmax(P, -1)
    return P @ V


# def multi_head(
#     Q: Float[Tensor, " ... queries hd_k"], K: Float[Tensor, " ... keys hd_k"], V: Float[Tensor, " ... values hd_v"],
#     mask: Bool[Tensor, " ... queries keys"] | None = None,
# ) -> Float[Tensor, " ... queries hd_v"]:
#     pass


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: int = 10000,
        max_seq_len: int = 1000,
        token_positions=None,
        pos_encode=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.token_positions = token_positions
        self.kwargs = {"device": device, "dtype": dtype}
        self.pos_encode = pos_encode

        self.W_Q = Linear(d_model, d_model, **self.kwargs)
        self.W_K = Linear(d_model, d_model, **self.kwargs)
        self.W_V = Linear(d_model, d_model, **self.kwargs)
        self.W_O = Linear(d_model, d_model, **self.kwargs)
        self.rope = RoPE(rope_theta, self.d_k, max_seq_len, device)

    def multi_head(
        self,
        Q: Float[Tensor, " ... seq_len d_model"],
        K: Float[Tensor, " ... seq_len d_model"],
        V: Float[Tensor, " ... seq_len d_model"],
    ) -> Float[Tensor, " ... values d_model"]:
        # print(Q.shape)
        # print(Q.view(*Q.shape[:-1], self.num_heads, self.d_k).shape)
        Q: Float[Tensor, " ... num_heads seq_len d_k"] = Q.view(*Q.shape[:-1], self.num_heads, self.d_k).transpose(
            -2, -3
        )
        K: Float[Tensor, " ... num_heads seq_len d_k"] = K.view(*K.shape[:-1], self.num_heads, self.d_k).transpose(
            -2, -3
        )
        V: Float[Tensor, " ... num_heads seq_len d_k"] = V.view(*V.shape[:-1], self.num_heads, self.d_k).transpose(
            -2, -3
        )

        prev_shape = Q.shape[:-2]
        seq_len = Q.shape[-2]
        new_shape = [1] * len(prev_shape)
        new_shape.append(seq_len)
        mask = torch.tril(torch.ones([seq_len, seq_len], dtype=torch.bool, device=Q.device))

        if self.pos_encode:
            if self.token_positions is None:
                self.token_positions = torch.arange(seq_len).view(new_shape).expand(*prev_shape, seq_len)

            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)

        out = rearrange(
            scaled_dot_product_attention(Q, K, V, mask),
            "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",
        )
        return out

    def forward(self, x: Float[Tensor, " ... seq_len d_model"]) -> Tensor:
        Q: Tensor = self.W_Q(x)
        K: Tensor = self.W_K(x)
        V: Tensor = self.W_V(x)

        return self.W_O(self.multi_head(Q, K, V))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.kwargs = {"device": device, "dtype": dtype}

        self.attention = MultiHeadSelfAttention(d_model, num_heads, **self.kwargs)
        self.feed_forward = SwiGLU(d_model, d_ff, **self.kwargs)
        self.norm1 = RMSNorm(d_model, **self.kwargs)
        self.norm2 = RMSNorm(d_model, **self.kwargs)

    def forward(self, x: Float[Tensor, " ... seq_len d_model"]):
        x: Tensor = x + self.attention(self.norm1(x))
        x: Tensor = x + self.feed_forward(self.norm2(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.kwargs = {"device": device, "dtype": dtype}
        self.embedding = Embedding(vocab_size, d_model, **self.kwargs)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len=context_length, **self.kwargs)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, **self.kwargs)
        self.lm_head = Linear(d_model, vocab_size, **self.kwargs)

    def forward(self, x: Int[Tensor, " ... seq_len"]) -> Float[Tensor, " ... seq_len vocab_size"]:
        x: Float[Tensor, " ... seq_len d_model"] = self.embedding(x)

        x: Float[Tensor, " ... seq_len d_model"] = self.transformer_blocks(x)

        x = self.lm_head(self.ln_final(x))
        # x = softmax(x, -1)

        return x
