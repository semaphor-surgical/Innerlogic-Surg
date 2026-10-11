# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Inference-only re-implementation of the DINOv3 ViT backbone, derived from
# https://github.com/facebookresearch/dinov3. Parameter and buffer names match
# the official layout so that DINOv3-format checkpoints load with strict=True.

import math
from functools import partial
from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    """qkv projection whose bias is multiplied by a fixed mask (zero on the K third)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        assert out_features % 3 == 0
        if bias:
            mask = torch.ones_like(self.bias)
            mask[out_features // 3 : 2 * out_features // 3] = 0
            self.register_buffer("bias_mask", mask)

    def forward(self, input: Tensor) -> Tensor:
        bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, bias)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_values)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), (h, w)


class RopePositionEmbedding(nn.Module):
    """Axial 2D RoPE with no learnable weights; `periods` is a persistent buffer."""

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        d_head = embed_dim // num_heads
        periods = base ** (2 * torch.arange(d_head // 4, dtype=torch.float32) / (d_head // 2))
        self.register_buffer("periods", periods, persistent=True)

    def forward(self, *, H: int, W: int) -> Tuple[Tensor, Tensor]:
        dd = {"device": self.periods.device, "dtype": torch.float32}
        coords_h = torch.arange(0.5, H, **dd) / H
        coords_w = torch.arange(0.5, W, **dd) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods.float()[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True, align_to: int = 8):
        super().__init__()
        d = int(hidden_features * 2 / 3)
        hidden = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, hidden, bias=bias)
        self.w2 = nn.Linear(in_features, hidden, bias=bias)
        self.w3 = nn.Linear(hidden, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


FFN_LAYERS = {"mlp": Mlp, "swiglu": SwiGLUFFN}


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, proj_bias: bool, mask_k_bias: bool):
        super().__init__()
        self.num_heads = num_heads
        qkv_cls = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = qkv_cls(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    @staticmethod
    def _apply_rope(q: Tensor, k: Tensor, rope: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # RoPE is applied to patch tokens only; CLS and storage tokens are left untouched.
        sin, cos = rope
        q_dtype, k_dtype = q.dtype, k.dtype
        q, k = q.to(sin.dtype), k.to(sin.dtype)
        prefix = q.shape[-2] - sin.shape[-2]
        q = torch.cat((q[:, :, :prefix], _rope_apply(q[:, :, prefix:], sin, cos)), dim=-2)
        k = torch.cat((k[:, :, :prefix], _rope_apply(k[:, :, prefix:], sin, cos)), dim=-2)
        return q.to(q_dtype), k.to(k_dtype)

    def forward(self, x: Tensor, rope: Tuple[Tensor, Tensor]) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = (t.transpose(1, 2) for t in torch.unbind(qkv, 2))
        q, k = self._apply_rope(q, k, rope)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        layerscale: bool,
        norm_layer,
        ffn_layer: str,
        mask_k_bias: bool,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, proj_bias, mask_k_bias)
        self.ls1 = LayerScale(dim) if layerscale else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FFN_LAYERS[ffn_layer](dim, int(dim * ffn_ratio), bias=ffn_bias)
        self.ls2 = LayerScale(dim) if layerscale else nn.Identity()

    def forward(self, x: Tensor, rope: Tuple[Tensor, Tensor]) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int = 16,
        in_chans: int = 3,
        ffn_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        layerscale: bool = True,
        n_storage_tokens: int = 4,
        mask_k_bias: bool = True,
        untie_cls_and_patch_norms: bool = False,
        norm_eps: float = 1e-5,
        rope_base: float = 100.0,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.embed_dim = self.num_features = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, embed_dim))
        self.rope_embed = RopePositionEmbedding(embed_dim, num_heads, base=rope_base)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim, num_heads, ffn_ratio, qkv_bias, proj_bias, ffn_bias,
                    layerscale, norm_layer, ffn_layer, mask_k_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm = norm_layer(embed_dim) if untie_cls_and_patch_norms else None
        # Unused at inference; kept so DINOv3-format checkpoints load with strict=True.
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def _tokens(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError(f"expected a (B, C, H, W) tensor, got shape {tuple(x.shape)}")
        H, W = x.shape[-2:]
        p = self.patch_size
        if H % p or W % p:
            raise ValueError(f"image size {H}x{W} must be a multiple of the patch size {p}")
        x, hw = self.patch_embed(x)
        B = x.shape[0]
        prefix = [self.cls_token.expand(B, -1, -1)]
        if self.n_storage_tokens > 0:
            prefix.append(self.storage_tokens.expand(B, -1, -1))
        return torch.cat(prefix + [x], dim=1), hw

    def _normalize(self, out: Tensor) -> Tensor:
        n_prefix = self.n_storage_tokens + 1
        if self.untie_cls_and_patch_norms:
            return torch.cat((self.cls_norm(out[:, :n_prefix]), self.norm(out[:, n_prefix:])), dim=1)
        return self.norm(out)

    def _run_blocks(self, x: Tensor, take: Sequence[int]) -> list:
        x, (h, w) = self._tokens(x)
        rope = self.rope_embed(H=h, W=w)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope)
            if i in take:
                outputs.append(x)
        return outputs

    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ):
        """Same contract as the official DINOv3 method: patch tokens of the last `n` blocks,
        optionally paired with the CLS token and/or the storage tokens."""
        take = range(self.n_blocks - n, self.n_blocks) if isinstance(n, int) else n
        outputs = self._run_blocks(x, take)
        if norm:
            outputs = [self._normalize(o) for o in outputs]
        cls_tokens = [o[:, 0] for o in outputs]
        extra_tokens = [o[:, 1 : self.n_storage_tokens + 1] for o in outputs]
        patches = [o[:, self.n_storage_tokens + 1 :] for o in outputs]
        if reshape:
            B, _, h, w = x.shape
            patches = [
                p.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for p in patches
            ]
        if return_class_token and return_extra_tokens:
            return tuple(zip(patches, cls_tokens, extra_tokens))
        if return_class_token:
            return tuple(zip(patches, cls_tokens))
        if return_extra_tokens:
            return tuple(zip(patches, extra_tokens))
        return tuple(patches)

    def forward_features(self, x: Tensor) -> dict:
        (out,) = self._run_blocks(x, [self.n_blocks - 1])
        out = self._normalize(out)
        n_prefix = self.n_storage_tokens + 1
        return {
            "x_norm_clstoken": out[:, 0],
            "x_storage_tokens": out[:, 1:n_prefix],
            "x_norm_patchtokens": out[:, n_prefix:],
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)["x_norm_clstoken"]
