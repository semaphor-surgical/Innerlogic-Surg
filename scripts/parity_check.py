#!/usr/bin/env python
"""Check that `surgfm_probe` reproduces the reference DINOv3 ViT implementation on a checkpoint.

    python scripts/parity_check.py --dinov3-repo /path/to/dinov3 --checkpoint weights.pth

`--dinov3-repo` is a checkout of the official DINOv3 repository (https://github.com/facebookresearch/dinov3).
The reference model is built from the hub configuration of the matching size, independently of
`surgfm_probe.backbone.infer_config`, and loaded with strict=True. Features from both
implementations are compared on real images (--images) or random noise.
"""

import argparse
import sys
from pathlib import Path

import torch

# Hub configuration of each size: (embed_dim, depth, num_heads, ffn_layer, ffn_ratio)
HUB = {
    384: (384, 12, 6, "mlp", 4.0),
    768: (768, 12, 12, "mlp", 4.0),
    1024: (1024, 24, 16, "mlp", 4.0),
    1280: (1280, 32, 20, "swiglu", 6.0),
}


def build_reference(repo: Path, embed_dim: int):
    sys.path.insert(0, str(repo))
    from dinov3.models.vision_transformer import DinoVisionTransformer

    dim, depth, heads, ffn_layer, ffn_ratio = HUB[embed_dim]
    return DinoVisionTransformer(
        img_size=224, patch_size=16, in_chans=3,
        pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32",
        embed_dim=dim, depth=depth, num_heads=heads, ffn_ratio=ffn_ratio, qkv_bias=True,
        drop_path_rate=0.0, layerscale_init=1.0e-05, norm_layer="layernormbf16",
        ffn_layer=ffn_layer, ffn_bias=True, proj_bias=True, n_storage_tokens=4, mask_k_bias=True,
    )  # fmt: skip


def load_images(folder, n, size=224):
    from PIL import Image

    from surgfm_probe import eval_transform

    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:n]
    if not paths:
        raise SystemExit(f"no images in {folder}")
    tf = eval_transform(256, size)
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dinov3-repo", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--images", help="folder of real images (default: random noise)")
    ap.add_argument("--n", type=int, default=16, help="number of images")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--atol", type=float, default=2e-4, help="max |difference| allowed in fp32")
    args = ap.parse_args()

    from surgfm_probe import FeatureExtractor, load_backbone

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ref = build_reference(args.dinov3_repo, sd["cls_token"].shape[-1])
    ref.load_state_dict(sd, strict=True)
    ref = ref.to(args.device).eval()
    ours = load_backbone(args.checkpoint, device=args.device)

    x = load_images(args.images, args.n) if args.images else torch.randn(args.n, 3, 224, 224)
    x = x.to(args.device)
    x_wide = torch.randn(2, 3, 160, 288, device=args.device)  # non-square input exercises RoPE

    ok = True
    with torch.no_grad():
        for name, inp in (("224x224", x), ("160x288", x_wide)):
            (rp, rc), (op, oc) = (
                ref.get_intermediate_layers(inp, n=1, return_class_token=True)[0],
                ours.get_intermediate_layers(inp, n=1, return_class_token=True)[0],
            )
            for what, a, b in (("cls", rc, oc), ("patch", rp, op)):
                diff = (a - b).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1)).min().item()
                good = diff <= args.atol
                ok &= good
                print(f"{name:8s} {what:5s} max|diff| {diff:.2e}  min cos {cos:.8f}  {'ok' if good else 'MISMATCH'}")

        fx = FeatureExtractor(ours)
        rc4 = ref.get_intermediate_layers(x, n=4, return_class_token=True)
        rfeat = torch.cat([c for _, c in rc4[-1:]], -1)
        rfeat = torch.cat([rfeat, rc4[-1][0].mean(1)], -1)
        diff = (fx(x) - rfeat).abs().max().item()
        ok &= diff <= args.atol
        print(f"descriptor (cls + mean patch) max|diff| {diff:.2e}  {'ok' if diff <= args.atol else 'MISMATCH'}")

        if args.device != "cpu":
            fb = FeatureExtractor(ours, autocast_dtype="bfloat16")(x)
            cos = torch.nn.functional.cosine_similarity(fb, fx(x)).min().item()
            print(f"bf16 autocast vs fp32 descriptor: min cos {cos:.6f}")

    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
