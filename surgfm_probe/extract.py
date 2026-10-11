"""Batch feature extraction from image folders, manifests and videos, plus a small `.npz` format."""

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import ImageDataset
from .features import FeatureExtractor


def _progress(iterable: Iterable, desc: str, total: Optional[int] = None):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def _device(device: Union[str, torch.device, None]) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@torch.no_grad()
def extract_dataset(
    extractor: FeatureExtractor,
    dataset: ImageDataset,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    device: Union[str, torch.device, None] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run the extractor over `dataset` in order. Returns (features (N, D) float32, labels or None)."""
    device = _device(device)
    extractor.to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feats, labels = [], []
    for images, y in _progress(loader, "extract", total=len(loader)):
        feats.append(extractor(images.to(device, non_blocking=True)).cpu())
        labels.append(y)
    features = torch.cat(feats).numpy()
    return features, (torch.cat(labels).numpy() if dataset.label_field else None)


def iter_video_frames(
    path: Union[str, Path], fps: float = 1.0, max_frames: Optional[int] = None
) -> Iterator[Tuple[int, float, Image.Image]]:
    """Yield ``(frame_index, time_in_seconds, RGB PIL image)`` roughly every ``native_fps / fps`` frames."""
    try:
        import cv2
    except ImportError:
        raise ImportError("video input needs OpenCV: pip install 'surgfm-probe[video]'") from None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        native_fps = 30.0
    step = max(1, round(native_fps / fps))
    index, emitted = 0, 0
    try:
        while max_frames is None or emitted < max_frames:
            if not cap.grab():
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield index, index / native_fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    emitted += 1
            index += 1
    finally:
        cap.release()


@torch.no_grad()
def extract_video(
    extractor: FeatureExtractor,
    path: Union[str, Path],
    transform: Callable,
    *,
    fps: float = 1.0,
    max_frames: Optional[int] = None,
    batch_size: int = 64,
    device: Union[str, torch.device, None] = None,
) -> dict:
    """Per-frame features for a video. Returns features, frame_index and time_s arrays."""
    device = _device(device)
    extractor.to(device)
    feats, frame_idx, times, batch = [], [], [], []

    def flush():
        if batch:
            feats.append(extractor(torch.stack(batch).to(device)).cpu())
            batch.clear()

    for idx, t, image in _progress(iter_video_frames(path, fps, max_frames), "frames"):
        batch.append(transform(image))
        frame_idx.append(idx)
        times.append(t)
        if len(batch) == batch_size:
            flush()
    flush()
    if not feats:
        raise ValueError(f"no frames decoded from {path}")
    return {
        "features": torch.cat(feats).numpy(),
        "frame_index": np.asarray(frame_idx, dtype=np.int64),
        "time_s": np.asarray(times, dtype=np.float64),
    }


def save_features(path: Union[str, Path], features: np.ndarray, meta: Optional[dict] = None, **arrays) -> None:
    """Write a `.npz` with `features`, a JSON `meta` string and any extra arrays (paths, labels, ...)."""
    arrays = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=np.asarray(features, dtype=np.float32), meta=json.dumps(meta or {}), **arrays)


def load_features(path: Union[str, Path]) -> dict:
    """Inverse of `save_features`; `meta` is returned as a dict."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"])) if "meta" in out else {}
    return out
