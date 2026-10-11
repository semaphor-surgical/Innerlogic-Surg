import json

import pytest
import torch

from surgfm_probe.data import ImageDataset, list_images, parse_label, read_manifest, resolve_field


def test_manifest_formats(tmp_path):
    rows = [{"image": "a.png", "phase": {"id": 2}}, {"image": "b.png", "phase": {"id": 5}}]
    (tmp_path / "m.json").write_text(json.dumps(rows))
    (tmp_path / "m.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "m.csv").write_text("image,phase.id\na.png,2\nb.png,5\n")
    assert read_manifest(tmp_path / "m.json") == rows
    assert read_manifest(tmp_path / "m.jsonl") == rows
    csv_rows = read_manifest(tmp_path / "m.csv")
    assert [resolve_field(r, "phase.id") for r in csv_rows] == ["2", "5"]  # flat CSV column
    assert [resolve_field(r, "phase.id") for r in rows] == [2, 5]  # nested JSON
    assert resolve_field(rows[0], "phase.missing") is None


def test_parse_label():
    assert parse_label("3", False, 5) == 3
    assert parse_label([0, 2], True, 4).tolist() == [1, 0, 1, 0]
    assert parse_label("1;3", True, 4).tolist() == [0, 1, 0, 1]
    assert parse_label("", True, 3).tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        parse_label([0], True, None)


def test_dataset(image_dir):
    root, rows = image_dir
    assert len(list_images(root)) == 12
    ds = ImageDataset(rows, root, lambda im: torch.tensor(im.size), "label.id")
    img, y = ds[4]
    assert y == rows[4]["label"]["id"] and img.tolist() == [128, 96]
    unlabeled = ImageDataset(list_images(root), transform=lambda im: torch.tensor(0))
    assert unlabeled[0][1] == -1


def test_missing_label_is_reported(image_dir):
    root, rows = image_dir
    ds = ImageDataset([{"image": rows[0]["image"]}], root, lambda im: 0, "label.id")
    with pytest.raises(ValueError, match="label field"):
        ds[0]
