import json

import numpy as np
import pytest
import torch

from surgfm_probe import FeatureExtractor, ImageDataset, eval_transform, extract_dataset, load_backbone
from surgfm_probe.cli import main
from surgfm_probe.extract import extract_video, iter_video_frames, load_features, save_features


def test_extract_matches_direct_forward(tiny_checkpoint, image_dir):
    root, rows = image_dir
    fx = FeatureExtractor(load_backbone(tiny_checkpoint))
    ds = ImageDataset(rows, root, eval_transform(), "label.id")
    feats, labels = extract_dataset(fx, ds, batch_size=5, num_workers=0, device="cpu")
    assert feats.shape == (12, 128) and labels.tolist() == [r["label"]["id"] for r in rows]
    x = torch.stack([ds[i][0] for i in (3, 7)])
    np.testing.assert_allclose(feats[[3, 7]], fx(x).numpy(), rtol=1e-4, atol=1e-5)


def test_npz_roundtrip(tmp_path):
    save_features(tmp_path / "f.npz", np.ones((3, 4)), {"a": 1}, paths=np.array(["x", "y", "z"]), labels=None)
    z = load_features(tmp_path / "f.npz")
    assert z["features"].dtype == np.float32 and z["meta"] == {"a": 1}
    assert z["paths"].tolist() == ["x", "y", "z"] and "labels" not in z


def _write_video(path, n_frames=90, fps=30):
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 128))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write MJPG")
    for i in range(n_frames):
        writer.write(np.full((128, 160, 3), (i * 2) % 255, dtype=np.uint8))
    writer.release()


def test_video_sampling(tmp_path, tiny_checkpoint):
    video = tmp_path / "v.avi"
    _write_video(video)
    frames = list(iter_video_frames(video, fps=1.0))
    assert [f[0] for f in frames] == [0, 30, 60]
    assert [round(f[1], 2) for f in frames] == [0.0, 1.0, 2.0]
    fx = FeatureExtractor(load_backbone(tiny_checkpoint))
    out = extract_video(fx, video, eval_transform(), fps=2.0, device="cpu")
    assert out["features"].shape == (6, 128) and out["frame_index"].tolist() == [0, 15, 30, 45, 60, 75]
    assert len(list(iter_video_frames(video, fps=1.0, max_frames=2))) == 2


def test_cli_extract_then_cached_probe(tmp_path, tiny_checkpoint, image_dir, capsys):
    root, rows = image_dir
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    common = ["--checkpoint", str(tiny_checkpoint), "--device", "cpu", "--num-workers", "0", "--batch-size", "4"]
    for split in ("train", "val", "test"):
        main(["extract", *common, "--manifest", str(manifest), "--root", str(root),
              "--label-field", "label.id", "--out", str(tmp_path / f"{split}.npz")])
    z = load_features(tmp_path / "train.npz")
    assert z["features"].shape == (12, 128) and z["labels"].shape == (12,)
    assert z["meta"]["feature_dim"] == 128 and z["meta"]["checkpoint"] == "tiny.pth"

    def probe(val, out):
        main(["probe", "--train-features", str(tmp_path / "train.npz"), "--val-features", str(tmp_path / val),
              "--test-features", str(tmp_path / "test.npz"), "--epochs", "1", "--epoch-length", "20",
              "--batch-size", "8", "--device", "cpu", "--output-dir", str(tmp_path / out)])
        return json.loads((tmp_path / out / "results.json").read_text())

    r = probe("val.npz", "separate")
    assert r["num_classes"] == 3 and r["protocol"] == "cached" and r["selection_split"] == "val"
    capsys.readouterr()
    r = probe("test.npz", "same")  # identical val and test files are detected and labelled
    assert r["selection_split"] == "test"
    assert "optimistic" in capsys.readouterr().out


def test_cli_standard_probe_on_images(tmp_path, tiny_checkpoint, image_dir):
    root, rows = image_dir
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(rows))
    main(["probe", "--checkpoint", str(tiny_checkpoint), "--train", str(manifest), "--val", str(manifest),
          "--test", str(manifest), "--root", str(root), "--label-field", "label.id", "--num-classes", "3",
          "--epochs", "1", "--epoch-length", "5", "--batch-size", "4", "--num-workers", "0",
          "--device", "cpu", "--output-dir", str(tmp_path / "std")])
    r = json.loads((tmp_path / "std" / "results.json").read_text())
    assert r["protocol"] == "standard" and r["train_augmentation"] and r["selection_split"] == "test"
    assert (tmp_path / "std" / "best_head.pt").exists()


def test_cli_requires_explicit_test_selection(tmp_path, tiny_checkpoint, image_dir):
    root, rows = image_dir
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(rows))
    with pytest.raises(SystemExit, match="select-on-test"):
        main(["probe", "--checkpoint", str(tiny_checkpoint), "--train", str(manifest), "--test", str(manifest),
              "--root", str(root), "--label-field", "label.id", "--num-classes", "3",
              "--device", "cpu", "--output-dir", str(tmp_path / "x")])
