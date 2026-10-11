import numpy as np
import pytest

from surgfm_probe.metrics import classification_metrics

sk = pytest.importorskip("sklearn.metrics")


@pytest.mark.parametrize("seed", range(5))
def test_single_label_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 6, 300)
    p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 8, 300))  # also predicts unseen classes
    acc, f1m, f1w = classification_metrics(y, p, num_classes=8)
    assert acc == pytest.approx(sk.accuracy_score(y, p))
    assert f1m == pytest.approx(sk.f1_score(y, p, average="macro", zero_division=0))
    assert f1w == pytest.approx(sk.f1_score(y, p, average="weighted", zero_division=0))


@pytest.mark.parametrize("seed", range(5))
def test_multi_label_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random((200, 7)) < 0.3).astype(int)
    y[:, 6] = 0  # a class with no positives
    p = np.where(rng.random(y.shape) < 0.7, y, rng.integers(0, 2, y.shape))
    primary, f1m, f1w = classification_metrics(y, p, multi_label=True)
    assert primary == f1m == pytest.approx(sk.f1_score(y, p, average="macro", zero_division=0))
    assert f1w == pytest.approx(sk.f1_score(y, p, average="weighted", zero_division=0))
