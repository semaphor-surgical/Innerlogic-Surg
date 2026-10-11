"""Classification metrics (numpy only; verified against scikit-learn in the test suite)."""

from typing import Tuple

import numpy as np


def _safe_f1(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    denom = n_pred + n_true
    return np.divide(2.0 * tp, denom, out=np.zeros(tp.shape, dtype=np.float64), where=denom > 0)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, multi_label: bool = False, num_classes: int = 0
) -> Tuple[float, float, float]:
    """Return ``(primary, f1_macro, f1_weighted)``.

    Single-label: ``primary`` is accuracy; F1 averages over the classes that occur in the labels
    or the predictions (scikit-learn's default). Multi-label: inputs are ``(N, C)`` 0/1 arrays,
    ``primary`` is macro-F1 over all C columns.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if multi_label:
        yt, yp = y_true.astype(bool), y_pred.astype(bool)
        tp = (yt & yp).sum(0).astype(np.float64)
        n_pred, n_true = yp.sum(0).astype(np.float64), yt.sum(0).astype(np.float64)
        f1 = _safe_f1(tp, n_pred, n_true)
        macro = float(f1.mean())
        weighted = float((f1 * n_true).sum() / n_true.sum()) if n_true.sum() > 0 else 0.0
        return macro, macro, weighted

    n = max(int(num_classes), int(y_true.max(initial=0)) + 1, int(y_pred.max(initial=0)) + 1)
    tp = np.bincount(y_true[y_true == y_pred], minlength=n).astype(np.float64)
    n_pred = np.bincount(y_pred, minlength=n).astype(np.float64)
    n_true = np.bincount(y_true, minlength=n).astype(np.float64)
    f1 = _safe_f1(tp, n_pred, n_true)
    present = (n_true > 0) | (n_pred > 0)
    macro = float(f1[present].mean()) if present.any() else 0.0
    weighted = float((f1 * n_true).sum() / n_true.sum()) if n_true.sum() > 0 else 0.0
    return float((y_true == y_pred).mean()), macro, weighted
