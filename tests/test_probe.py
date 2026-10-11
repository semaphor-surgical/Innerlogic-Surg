import json

import numpy as np
import pytest
import torch

from surgfm_probe import probe_features
from surgfm_probe.probe import DEFAULT_LR_GRID, HeadBank

FAST = dict(epochs=2, epoch_length=60, batch_size=32, device="cpu")


def blobs(n, classes=4, dim=16, seed=0, noise=0.6):
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(99).normal(size=(classes, dim)) * 2
    y = rng.integers(0, classes, n)
    return (centers[y] + rng.normal(size=(n, dim)) * noise).astype(np.float32), y


def test_cached_probe_learns_and_reports(tmp_path):
    out = probe_features(blobs(400), blobs(120, seed=1), blobs(120, seed=2), num_classes=4,
                         output_dir=tmp_path, **FAST)
    r = out.results
    assert r["test_accuracy"] > 90 and r["best_val_accuracy"] > 90
    assert r["protocol"] == "cached" and r["selection_split"] == "val"
    assert r["primary_metric"] == "accuracy" and r["feature_dim"] == 16
    assert (r["n_train"], r["n_val"], r["n_test"]) == (400, 120, 120)
    assert len(r["lr_grid"]) == 13 and r["best_classifier"].startswith("lr_")
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["test_accuracy"] == r["test_accuracy"]
    head = torch.load(tmp_path / "best_head.pt", weights_only=True)
    assert head["state_dict"]["weight"].shape == (4, 16)


def test_same_seed_is_reproducible():
    args = (blobs(300), blobs(100, seed=1), blobs(100, seed=2))
    a = probe_features(*args, num_classes=4, seed=7, **FAST).results
    b = probe_features(*args, num_classes=4, seed=7, **FAST).results
    assert a["test_accuracy"] == b["test_accuracy"] and a["best_classifier"] == b["best_classifier"]


def test_selecting_on_test_must_be_explicit():
    train, test = blobs(200), blobs(80, seed=2)
    with pytest.raises(ValueError, match="select_on_test"):
        probe_features(train, None, test, num_classes=4, **FAST)
    with pytest.raises(ValueError, match="select_on_test"):
        probe_features(train, test, test, num_classes=4, **FAST)  # same object
    r = probe_features(train, None, test, num_classes=4, select_on_test=True, **FAST).results
    assert r["selection_split"] == "test"
    assert r["best_val_accuracy"] == r["test_accuracy"]


def test_no_test_split():
    r = probe_features(blobs(200), blobs(80, seed=1), None, num_classes=4, **FAST).results
    assert r["test_accuracy"] is None and r["n_test"] is None


def test_label_checks():
    X, y = blobs(100)
    with pytest.raises(ValueError, match="num_classes"):
        probe_features((X, y), (X, y), num_classes=3, **FAST)
    with pytest.raises(ValueError, match="no label field"):
        probe_features((X, np.full(100, -1)), (X, y), num_classes=4, **FAST)


def test_multi_label():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(12, 5))

    def make(n, seed):
        X = np.random.default_rng(seed).normal(size=(n, 12)).astype(np.float32)
        return X, ((X @ W) > 0.5).astype(np.float32)

    r = probe_features(make(500, 0), make(150, 1), make(150, 2), num_classes=5, multi_label=True,
                       **{**FAST, "epochs": 3}).results
    assert r["primary_metric"] == "f1_macro" and r["multi_label"]
    assert r["test_f1_macro"] > 50 and r["test_accuracy"] == r["test_f1_macro"]


def test_tiny_train_split_smaller_than_batch():
    r = probe_features(blobs(20), blobs(30, seed=1), blobs(30, seed=2), num_classes=4,
                       **{**FAST, "batch_size": 64}).results
    assert r["n_train"] == 20 and r["test_accuracy"] is not None


def test_head_bank_lr_scaling_and_keys():
    bank = HeadBank(8, 3, DEFAULT_LR_GRID, 256)
    assert list(bank.lrs.values()) == list(DEFAULT_LR_GRID)
    assert "lr_0_001000" in bank.lrs
    small = HeadBank(8, 3, DEFAULT_LR_GRID, 8)  # scaled rates round together at 6 decimals
    assert len(small.lrs) == 13
    with pytest.raises(ValueError, match="duplicate"):
        HeadBank(8, 3, (0.1, 0.1), 256)
