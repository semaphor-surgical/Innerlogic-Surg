"""Manifest readers and image datasets.

A manifest lists images (and optionally labels). Accepted formats:

* JSON  - a list of objects, e.g. ``[{"image": "a/1.png", "phase": {"id": 3}}, ...]``
* JSONL - one such object per line
* CSV   - a header row with an ``image`` column and, for labelled data, a label column

``image`` paths are resolved against ``root``. Labels are read with a dotted path
(``--label-field phase.id``). A multi-label sample carries a list of active class indices
(JSON) or ``;``-separated indices (CSV).
"""

import csv
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def read_manifest(path: Union[str, Path]) -> List[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    text = path.read_text().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def resolve_field(sample: dict, field: str) -> Any:
    """Look up a dotted path such as ``phase.id``; returns None if any part is missing."""
    if field in sample:  # flat key, e.g. a CSV column literally named "phase.id"
        return sample[field]
    value: Any = sample
    for part in field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def parse_label(raw: Any, multi_label: bool, num_classes: Optional[int]) -> Union[int, torch.Tensor]:
    if multi_label:
        if num_classes is None:
            raise ValueError("num_classes is required for multi-label data")
        if isinstance(raw, str):
            raw = [t for t in raw.replace(",", ";").replace(" ", ";").split(";") if t != ""]
        vec = torch.zeros(num_classes, dtype=torch.float32)
        for idx in raw:
            vec[int(idx)] = 1.0
        return vec
    return int(raw)


def list_images(directory: Union[str, Path]) -> List[Path]:
    """All images under `directory`, recursively, in sorted order."""
    directory = Path(directory)
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


class ImageDataset(Dataset):
    """Images from a manifest (`samples` + `root`) or from explicit paths.

    Yields ``(image, label)``; without a `label_field`, the label is -1.
    """

    def __init__(
        self,
        samples: Sequence[Union[dict, str, Path]],
        root: Union[str, Path, None] = None,
        transform: Optional[Callable] = None,
        label_field: Optional[str] = None,
        multi_label: bool = False,
        num_classes: Optional[int] = None,
    ):
        self.samples = list(samples)
        self.root = Path(root) if root else None
        self.transform = transform
        self.label_field = label_field
        self.multi_label = multi_label
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.samples)

    def image_path(self, index: int) -> Path:
        sample = self.samples[index]
        image = sample["image"] if isinstance(sample, dict) else sample
        if isinstance(image, list):  # multi-frame entries: use the last frame
            image = image[-1]
        path = Path(image)
        return self.root / path if self.root and not path.is_absolute() else path

    def label(self, index: int):
        sample = self.samples[index]
        raw = resolve_field(sample, self.label_field) if isinstance(sample, dict) else None
        if raw is None:
            raise ValueError(
                f"label field {self.label_field!r} not found in sample {index} "
                f"({sample.get('id', sample) if isinstance(sample, dict) else sample})"
            )
        return parse_label(raw, self.multi_label, self.num_classes)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        image = Image.open(self.image_path(index)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, (self.label(index) if self.label_field else -1)
