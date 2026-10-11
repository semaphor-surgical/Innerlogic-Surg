import pytest
import torch

from surgfm_probe import FeatureExtractor, load_backbone
from surgfm_probe.backbone import ARCHS, infer_config
from surgfm_probe.models import DinoVisionTransformer

from conftest import randomize, tiny_vit


def test_intermediate_layers_contract():
    m = tiny_vit()
    x = torch.randn(2, 3, 224, 224)
    out = m.get_intermediate_layers(x, n=2, return_class_token=True)
    assert len(out) == 2
    patches, cls = out[-1]
    assert patches.shape == (2, 196, 64) and cls.shape == (2, 64)
    (grid,) = m.get_intermediate_layers(x, n=1, reshape=True)
    assert grid.shape == (2, 64, 14, 14)


def test_non_square_and_bad_size():
    m = tiny_vit()
    (p,) = m.get_intermediate_layers(torch.randn(1, 3, 160, 224), n=1)
    assert p.shape == (1, 10 * 14, 64)
    with pytest.raises(ValueError):
        m.get_intermediate_layers(torch.randn(1, 3, 225, 224), n=1)


def test_state_dict_roundtrip_and_inference(tiny_checkpoint):
    m = load_backbone(tiny_checkpoint)
    ref = tiny_vit()
    x = torch.randn(2, 3, 224, 224)
    torch.testing.assert_close(m(x), ref(x))
    cfg = infer_config(ref.state_dict())
    assert (cfg["embed_dim"], cfg["depth"], cfg["num_heads"], cfg["patch_size"]) == (64, 3, 2, 16)
    assert cfg["n_storage_tokens"] == 4 and cfg["mask_k_bias"] and cfg["layerscale"]
    assert not any(p.requires_grad for p in m.parameters()) and not m.training


@pytest.mark.parametrize("arch", ["vit_small", "vit_base"])
def test_arch_presets_match_inference(arch):
    cfg = ARCHS[arch]
    sd = DinoVisionTransformer(**cfg).state_dict()
    inferred = infer_config(sd)
    assert {k: inferred[k] for k in cfg} == cfg


def test_swiglu_inference():
    m = tiny_vit(ffn_layer="swiglu", ffn_ratio=6.0)
    cfg = infer_config(m.state_dict())
    assert cfg["ffn_layer"] == "swiglu" and cfg["ffn_ratio"] == pytest.approx(6.0, rel=0.1)
    DinoVisionTransformer(**cfg).load_state_dict(m.state_dict(), strict=True)


def test_load_unwraps_checkpoint(tmp_path):
    m = tiny_vit()
    path = tmp_path / "wrapped.pth"
    torch.save({"teacher": {f"backbone.{k}": v for k, v in m.state_dict().items()}}, path)
    x = torch.randn(1, 3, 224, 224)
    torch.testing.assert_close(load_backbone(path)(x), m(x))


def test_not_a_backbone_is_rejected(tmp_path):
    path = tmp_path / "junk.pth"
    torch.save({"foo": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="does not look like"):
        load_backbone(path)


@pytest.mark.parametrize("n,avg,expected", [(1, True, 128), (1, False, 64), (4, True, 320), (2, False, 128)])
def test_feature_dim(n, avg, expected):
    m = tiny_vit(depth=4)
    fx = FeatureExtractor(m, n_last_blocks=n, use_avgpool=avg)
    out = fx(torch.randn(2, 3, 224, 224))
    assert fx.feature_dim == expected and out.shape == (2, expected) and out.dtype == torch.float32


def test_feature_definition():
    m = tiny_vit()
    x = torch.randn(2, 3, 224, 224)
    out = FeatureExtractor(m)(x)
    feats = m.forward_features(x)
    torch.testing.assert_close(out[:, :64], feats["x_norm_clstoken"])
    torch.testing.assert_close(out[:, 64:], feats["x_norm_patchtokens"].mean(1))


def test_dense_shape():
    assert FeatureExtractor(tiny_vit()).dense(torch.randn(1, 3, 224, 160)).shape == (1, 64, 14, 10)


def test_fp16_rejected_and_bf16_close():
    m = tiny_vit()
    with pytest.raises(ValueError, match="float16"):
        FeatureExtractor(m, autocast_dtype="float16")
    x = torch.randn(2, 3, 224, 224)
    ref = FeatureExtractor(m)(x)
    low = FeatureExtractor(m, autocast_dtype="bfloat16")(x)
    assert torch.nn.functional.cosine_similarity(ref, low).min() > 0.98
    assert low.dtype == torch.float32
