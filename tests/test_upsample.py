import pytest
import torch
import torch.nn.functional as F
from torch import nn

import surgfm_probe.upsample as upsample
from conftest import tiny_vit
from surgfm_probe import FeatureExtractor, load_anyup


class FakeUpsampler(nn.Module):
    """Same call signature as AnyUp; bilinear resize instead of the learned model."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image, features, output_size=None, q_chunk_size=None):
        self.calls.append((tuple(image.shape), tuple(features.shape), output_size, q_chunk_size))
        return F.interpolate(features, size=output_size or image.shape[-2:], mode="bilinear")


def test_dense_with_upsampler_defaults_to_input_resolution():
    extractor = FeatureExtractor(tiny_vit())
    x = torch.randn(2, 3, 64, 96)
    up = FakeUpsampler()
    out = extractor.dense(x, upsampler=up)
    assert out.shape == (2, 64, 64, 96) and out.dtype == torch.float32
    assert up.calls == [((2, 3, 64, 96), (2, 64, 4, 6), None, None)]
    assert extractor.dense(x).shape == (2, 64, 4, 6)


def test_dense_with_upsampler_output_size_and_chunking():
    extractor = FeatureExtractor(tiny_vit())
    up = FakeUpsampler()
    out = extractor.dense(torch.randn(1, 3, 64, 64), upsampler=up, output_size=(32, 48), q_chunk_size=8)
    assert out.shape == (1, 64, 32, 48)
    assert up.calls[0][2:] == ((32, 48), 8)


def test_load_anyup_selects_variant_and_freezes(monkeypatch):
    seen = {}

    def fake_hub_load(repo, entry, **kwargs):
        seen.update(repo=repo, entry=entry, kwargs=kwargs)
        return nn.Linear(2, 2).train()

    monkeypatch.setattr(upsample.torch.hub, "load", fake_hub_load)
    model = load_anyup(device="cpu", use_natten=False)
    assert seen["entry"] == "anyup_multi_backbone" and seen["repo"] == upsample.ANYUP_REPO
    assert seen["kwargs"] == {"trust_repo": True, "device": "cpu", "use_natten": False}
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    load_anyup("single_backbone")
    assert seen["entry"] == "anyup"
    with pytest.raises(ValueError, match="variant"):
        load_anyup("nope")
