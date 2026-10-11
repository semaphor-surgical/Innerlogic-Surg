"""Build a ViT backbone and load a DINOv3-format checkpoint into it."""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import nn

from .models.vit import DinoVisionTransformer

ARCHS: Dict[str, dict] = {
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12),
    "vit_large": dict(embed_dim=1024, depth=24, num_heads=16),
    "vit_huge2": dict(embed_dim=1280, depth=32, num_heads=20, ffn_ratio=6.0, ffn_layer="swiglu"),
}

_WRAPPER_KEYS = ("teacher", "student", "model", "state_dict")


def unwrap_state_dict(obj: dict) -> dict:
    """Return a flat backbone state dict from a plain or lightly wrapped checkpoint."""
    sd = obj
    for key in _WRAPPER_KEYS:
        if key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    sd = {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in sd.items()}
    if any(k.startswith("backbone.") for k in sd):
        sd = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
    return sd


def infer_config(sd: dict) -> dict:
    """Infer constructor arguments from tensor shapes and key names of a backbone state dict."""
    try:
        embed_dim = sd["cls_token"].shape[-1]
        patch_size = sd["patch_embed.proj.weight"].shape[-1]
    except KeyError as e:
        raise ValueError(
            f"checkpoint does not look like a DINOv3 ViT backbone (missing key {e}); "
            f"first keys: {list(sd)[:5]}"
        ) from None
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    if "rope_embed.periods" in sd:  # axial RoPE keeps head_dim / 4 frequencies
        num_heads = embed_dim // (4 * sd["rope_embed.periods"].numel())
    else:
        num_heads = next((a["num_heads"] for a in ARCHS.values() if a["embed_dim"] == embed_dim), embed_dim // 64)

    if "blocks.0.mlp.w1.weight" in sd:
        ffn_layer = "swiglu"
        ffn_ratio = sd["blocks.0.mlp.w1.weight"].shape[0] * 1.5 / embed_dim
    else:
        ffn_layer = "mlp"
        ffn_ratio = sd["blocks.0.mlp.fc1.weight"].shape[0] / embed_dim

    return dict(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        ffn_ratio=ffn_ratio,
        ffn_layer=ffn_layer,
        qkv_bias="blocks.0.attn.qkv.bias" in sd,
        proj_bias="blocks.0.attn.proj.bias" in sd,
        ffn_bias=("blocks.0.mlp.fc1.bias" in sd) or ("blocks.0.mlp.w1.bias" in sd),
        layerscale="blocks.0.ls1.gamma" in sd,
        n_storage_tokens=sd["storage_tokens"].shape[1] if "storage_tokens" in sd else 0,
        mask_k_bias="blocks.0.attn.qkv.bias_mask" in sd,
        untie_cls_and_patch_norms="cls_norm.weight" in sd,
    )


def load_backbone(
    checkpoint: Union[str, Path],
    arch: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> DinoVisionTransformer:
    """Load a frozen, eval-mode ViT backbone from a DINOv3-format `.pth` checkpoint.

    The architecture is inferred from the checkpoint unless `arch` names one of `ARCHS`.
    Parameters stay in float32; use `FeatureExtractor(..., autocast_dtype="bfloat16")`
    for reduced-precision inference.
    """
    obj = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    sd = unwrap_state_dict(obj)
    cfg = infer_config(sd)
    if arch is not None:
        if arch not in ARCHS:
            raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHS)}")
        cfg.update(ARCHS[arch])
    model = DinoVisionTransformer(**cfg)
    model.load_state_dict(sd, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def feature_dim(model: nn.Module, n_last_blocks: int = 1, use_avgpool: bool = True) -> int:
    return model.embed_dim * (n_last_blocks + (1 if use_avgpool else 0))
