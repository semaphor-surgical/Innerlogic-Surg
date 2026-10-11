"""Linear probing on a frozen backbone.

A bank of linear heads, one per learning rate, is trained side by side on identical batches.
The head with the best score on the selection split is evaluated on the test split.

Two protocols:

* ``standard`` - the backbone runs on every training step with random-resized-crop + flip
  augmentation (`probe_standard`). This is the protocol behind the reported tables.
* ``cached`` - features are extracted once with the evaluation transform and the heads are
  trained on the cache (`probe_features`). Much faster, but it drops augmentation and scores
  about 1-2 points lower; do not mix the two protocols in one table.
"""

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import ImageDataset
from .extract import extract_dataset
from .features import FeatureExtractor
from .metrics import classification_metrics

log = logging.getLogger("surgfm_probe")

DEFAULT_LR_GRID: Tuple[float, ...] = (
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 5e-2,
    0.1,
)  # fmt: skip

ArrayLike = Union[np.ndarray, Tensor]
Split = Tuple[ArrayLike, ArrayLike]  # (features (N, D), labels (N,) or multi-hot (N, C))


class ProbeResult(NamedTuple):
    results: dict
    head: nn.Linear


class HeadBank(nn.Module):
    """One linear head per learning rate. The learning rate is scaled by ``batch_size / 256``."""

    def __init__(self, in_dim: int, num_classes: int, lr_grid: Sequence[float], batch_size: int):
        super().__init__()
        if len(set(lr_grid)) != len(lr_grid):
            raise ValueError(f"lr_grid has duplicate values: {list(lr_grid)}")
        scaled = [lr * batch_size / 256.0 for lr in lr_grid]
        keys = [f"lr_{s:.6f}".replace(".", "_") for s in scaled]
        if len(set(keys)) != len(keys):  # tiny batch sizes round distinct rates together
            keys = [f"lr_{s:.3e}".replace(".", "_") for s in scaled]
        self.lrs: Dict[str, float] = dict(zip(keys, scaled))
        self.heads = nn.ModuleDict()
        for key in keys:
            head = nn.Linear(in_dim, num_classes)
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)
            self.heads[key] = head

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {k: h(x) for k, h in self.heads.items()}

    def param_groups(self):
        return [{"params": h.parameters(), "lr": self.lrs[k]} for k, h in self.heads.items()]


@dataclass
class _Split:
    X: Tensor
    Y: Tensor

    def __len__(self) -> int:
        return self.X.shape[0]


def _as_split(split: Split, num_classes: int, multi_label: bool, device: torch.device, name: str) -> _Split:
    X, Y = split
    X = torch.as_tensor(np.asarray(X) if not isinstance(X, Tensor) else X).float()
    Y = torch.as_tensor(np.asarray(Y) if not isinstance(Y, Tensor) else Y)
    if X.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"{name}: features {tuple(X.shape)} and labels {tuple(Y.shape)} do not match")
    if multi_label:
        if Y.ndim != 2 or Y.shape[1] != num_classes:
            raise ValueError(f"{name}: multi-label targets must have shape (N, {num_classes}), got {tuple(Y.shape)}")
        Y = Y.float()
    else:
        Y = Y.long()
        if Y.min() < 0 or Y.max() >= num_classes:
            raise ValueError(
                f"{name}: labels span [{int(Y.min())}, {int(Y.max())}] but num_classes={num_classes}"
                + (" (label -1 means the manifest has no label field)" if Y.min() < 0 else "")
            )
    return _Split(X.to(device), Y.to(device))


@torch.no_grad()
def _evaluate(bank: HeadBank, split: _Split, multi_label: bool, keys=None, full: bool = False) -> dict:
    """Per-head score on `split`: primary metric, or ``(primary, f1_macro, f1_weighted)`` if `full`."""
    bank.eval()
    out = {}
    y_np = split.Y.cpu().numpy() if (full or multi_label) else None
    for key in keys or bank.heads.keys():
        logits = bank.heads[key](split.X)
        if multi_label:
            pred = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.int64)
            scores = classification_metrics(y_np, pred, multi_label=True)
            out[key] = scores if full else scores[0]
        else:
            pred = logits.argmax(1)
            acc = (pred == split.Y).sum().item() / len(split)
            if full:
                f1m, f1w = classification_metrics(y_np, pred.cpu().numpy(), num_classes=logits.shape[1])[1:]
                out[key] = (acc, f1m, f1w)
            else:
                out[key] = acc
    return out


def _train(
    bank: HeadBank,
    batches: Callable[[int], Iterator[Tuple[Tensor, Tensor]]],
    *,
    epochs: int,
    epoch_length: int,
    optimizer: str,
    val: _Split,
    test: Optional[_Split],
    multi_label: bool,
) -> dict:
    if optimizer not in ("sgd", "adamw"):
        raise ValueError(f"optimizer must be 'sgd' or 'adamw', got {optimizer!r}")
    groups = bank.param_groups()
    opt = (
        torch.optim.AdamW(groups, weight_decay=0)
        if optimizer == "adamw"
        else torch.optim.SGD(groups, momentum=0.9, weight_decay=0)
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * epoch_length, eta_min=0)
    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

    best_val, best_key, best_state = -1.0, "", None
    epoch_metrics = []
    for epoch in range(epochs):
        bank.train()
        loss_sum = torch.zeros((), device=val.X.device)
        for xb, yb in batches(epoch):
            outs = bank(xb)
            loss = sum(criterion(o, yb) for o in outs.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach() / len(outs)
        scores = _evaluate(bank, val, multi_label)
        epoch_key = max(scores, key=scores.get)
        if scores[epoch_key] > best_val:
            best_val, best_key = scores[epoch_key], epoch_key
            best_state = copy.deepcopy(bank.heads[epoch_key].state_dict())
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(loss_sum.item() / epoch_length, 5),
                "best_val": round(scores[epoch_key] * 100, 2),
                "best_classifier": epoch_key,
            }
        )
        log.info(
            "epoch %d/%d  loss %.4f  val %.2f (%s)  best %.2f",
            epoch + 1, epochs, epoch_metrics[-1]["train_loss"], 100 * scores[epoch_key], epoch_key, 100 * best_val,
        )  # fmt: skip

    bank.heads[best_key].load_state_dict(best_state)
    val_acc, val_f1m, val_f1w = _evaluate(bank, val, multi_label, keys=[best_key], full=True)[best_key]
    test_scores = (None, None, None)
    if test is not None:
        test_scores = _evaluate(bank, test, multi_label, keys=[best_key], full=True)[best_key]

    def pct(v):
        return None if v is None else round(v * 100, 2)

    return {
        "best_val_accuracy": pct(best_val),
        "best_val_f1_macro": pct(val_f1m),
        "best_val_f1_weighted": pct(val_f1w),
        "best_classifier": best_key,
        "test_accuracy": pct(test_scores[0]),
        "test_f1_macro": pct(test_scores[1]),
        "test_f1_weighted": pct(test_scores[2]),
        "epochs": epochs,
        "epoch_length": epoch_length,
        "epoch_metrics": epoch_metrics,
    }


def _resolve_selection(val, test, select_on_test: bool):
    """Return ``(split_used_for_selection, name)``; selecting on the test split must be explicit."""
    if val is not None and val is not test:
        return val, "val"
    if test is None:
        raise ValueError("need a validation split, or a test split with select_on_test=True")
    if not select_on_test:
        raise ValueError(
            "no separate validation split: the best head would be picked on the test split. "
            "Pass select_on_test=True (CLI: --select-on-test) to accept this, "
            "and report the result as 'selected on test'."
        )
    return test, "test"


def _finalize(results: dict, *, protocol: str, selection_split: str, primary_metric: str, bank: HeadBank,
              extra: dict, start: float, seed: int, output_dir) -> ProbeResult:  # fmt: skip
    results.update(
        protocol=protocol,
        feature_cached=protocol == "cached",
        train_augmentation=protocol == "standard",
        selection_split=selection_split,
        primary_metric=primary_metric,
        lr_grid=list(bank.lrs.values()),
        seed=seed,
        training_time_seconds=int(time.time() - start),
        **extra,
    )
    head = bank.heads[results["best_classifier"]]
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                "classifier": results["best_classifier"],
                "in_dim": head.in_features,
                "num_classes": head.out_features,
                "multi_label": extra["multi_label"],
            },
            out / "best_head.pt",
        )
    return ProbeResult(results, head)


def _device(device) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def probe_features(
    train: Split,
    val: Optional[Split],
    test: Optional[Split] = None,
    *,
    num_classes: int,
    multi_label: bool = False,
    lr_grid: Sequence[float] = DEFAULT_LR_GRID,
    epochs: int = 10,
    epoch_length: int = 1250,
    batch_size: int = 256,
    optimizer: str = "sgd",
    seed: int = 1234,
    device: Union[str, torch.device, None] = None,
    output_dir: Union[str, Path, None] = None,
    select_on_test: bool = False,
) -> ProbeResult:
    """`cached` protocol: train the head bank on precomputed ``(features, labels)`` splits.

    Pass ``val=None, select_on_test=True`` when the benchmark has no separate validation split.
    """
    start = time.time()
    device = _device(device)
    torch.manual_seed(seed)
    val, selection_split = _resolve_selection(val, test, select_on_test)
    tr = _as_split(train, num_classes, multi_label, device, "train")
    va = _as_split(val, num_classes, multi_label, device, "val")
    te = va if val is test else (_as_split(test, num_classes, multi_label, device, "test") if test is not None else None)
    for name, s in (("val", va), ("test", te)):
        if s is not None and s.X.shape[1] != tr.X.shape[1]:
            raise ValueError(f"{name} feature dim {s.X.shape[1]} != train feature dim {tr.X.shape[1]}")

    bank = HeadBank(tr.X.shape[1], num_classes, lr_grid, batch_size).to(device)
    N = len(tr)
    offsets = torch.arange(batch_size, device=device)

    def batches(epoch: int):
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        perm = torch.randperm(N, generator=g).to(device)
        for step in range(epoch_length):
            idx = perm[(step * batch_size + offsets) % N]
            yield tr.X[idx], tr.Y[idx]

    results = _train(
        bank, batches, epochs=epochs, epoch_length=epoch_length, optimizer=optimizer,
        val=va, test=te, multi_label=multi_label,
    )  # fmt: skip
    extra = dict(
        feature_dim=tr.X.shape[1], num_classes=num_classes, multi_label=multi_label,
        n_train=len(tr), n_val=len(va), n_test=None if te is None else len(te), batch_size=batch_size,
        optimizer=optimizer,
    )  # fmt: skip
    return _finalize(
        results, protocol="cached", selection_split=selection_split,
        primary_metric="f1_macro" if multi_label else "accuracy",
        bank=bank, extra=extra, start=start, seed=seed, output_dir=output_dir,
    )  # fmt: skip


def probe_standard(
    extractor: FeatureExtractor,
    train: ImageDataset,
    val: Optional[ImageDataset],
    test: Optional[ImageDataset] = None,
    *,
    num_classes: int,
    multi_label: bool = False,
    lr_grid: Sequence[float] = DEFAULT_LR_GRID,
    epochs: int = 10,
    epoch_length: int = 1250,
    batch_size: int = 256,
    optimizer: str = "sgd",
    seed: int = 1234,
    num_workers: int = 8,
    device: Union[str, torch.device, None] = None,
    output_dir: Union[str, Path, None] = None,
    select_on_test: bool = False,
) -> ProbeResult:
    """`standard` protocol: the frozen backbone sees a fresh augmented view at every step.

    `train` should use `train_transform()`; `val` and `test` should use `eval_transform()`.
    The evaluation features do not change between epochs, so they are extracted once.
    """
    start = time.time()
    device = _device(device)
    torch.manual_seed(seed)
    val, selection_split = _resolve_selection(val, test, select_on_test)
    extractor.to(device)

    def features(ds: ImageDataset) -> _Split:
        X, Y = extract_dataset(extractor, ds, batch_size=batch_size, num_workers=num_workers, device=device)
        return _as_split((X, Y), num_classes, multi_label, device, "eval")

    va = features(val)
    te = va if val is test else (features(test) if test is not None else None)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train) >= batch_size,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    bank = HeadBank(extractor.feature_dim, num_classes, lr_grid, batch_size).to(device)
    it = iter(loader)

    def batches(epoch: int):
        nonlocal it
        for _ in range(epoch_length):
            try:
                images, labels = next(it)
            except StopIteration:
                it = iter(loader)
                images, labels = next(it)
            feats = extractor(images.to(device, non_blocking=True))
            labels = labels.float() if multi_label else labels.long()
            yield feats, labels.to(device, non_blocking=True)

    results = _train(
        bank, batches, epochs=epochs, epoch_length=epoch_length, optimizer=optimizer,
        val=va, test=te, multi_label=multi_label,
    )  # fmt: skip
    extra = dict(
        feature_dim=extractor.feature_dim, num_classes=num_classes, multi_label=multi_label,
        n_train=len(train), n_val=len(va), n_test=None if te is None else len(te), batch_size=batch_size,
        optimizer=optimizer,
    )  # fmt: skip
    return _finalize(
        results, protocol="standard", selection_split=selection_split,
        primary_metric="f1_macro" if multi_label else "accuracy",
        bank=bank, extra=extra, start=start, seed=seed, output_dir=output_dir,
    )  # fmt: skip
