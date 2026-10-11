import numpy as np
import pytest
import torch
from PIL import Image

from surgfm_probe.models import DinoVisionTransformer


def randomize(model: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gamma"):
                p.fill_(1.0)
            elif name.endswith("norm.weight"):
                p.fill_(1.0)
            elif name.endswith("norm.bias"):
                p.zero_()
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


def tiny_vit(**overrides) -> DinoVisionTransformer:
    cfg = dict(embed_dim=64, depth=3, num_heads=2, patch_size=16, ffn_ratio=2.0)
    cfg.update(overrides)
    return randomize(DinoVisionTransformer(**cfg)).eval()


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pth"
    torch.save(tiny_vit().state_dict(), path)
    return path


@pytest.fixture
def image_dir(tmp_path):
    """12 small solid-color-ish images in two class folders; returns (root, manifest rows)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "images"
    rows = []
    for i in range(12):
        label = i % 3
        arr = rng.integers(0, 60, size=(96, 128, 3), dtype=np.uint8)
        arr[..., label] += 150  # class signal in one color channel
        rel = f"c{label}/img{i:02d}.png"
        (root / f"c{label}").mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(root / rel)
        rows.append({"image": rel, "label": {"id": label}})
    return root, rows
