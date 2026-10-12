import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from surgfm_probe import load_features

ROOT = Path(__file__).resolve().parents[1]


def _run(script, *args):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script), *map(str, args)],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_extract_features_example(tmp_path, tiny_checkpoint, image_dir):
    root, _ = image_dir
    out = tmp_path / "f.npz"
    stdout = _run("extract_features.py", "--checkpoint", tiny_checkpoint, "--images", root, "--out", out)
    z = load_features(out)
    assert z["features"].shape == (12, 128) and len(z["paths"]) == 12
    assert "dense patch features" in stdout


def test_use_as_encoder_example(tiny_checkpoint):
    frozen = _run("use_as_encoder.py", "--checkpoint", tiny_checkpoint, "--num-classes", 3)
    full = _run("use_as_encoder.py", "--checkpoint", tiny_checkpoint, "--num-classes", 3, "--finetune")
    n_frozen = int(frozen.split("parameters:")[1].split()[0].replace(",", ""))
    n_full = int(full.split("parameters:")[1].split()[0].replace(",", ""))
    assert n_frozen == 64 * 3 + 3  # only the 1x1 conv head trains
    assert n_full > n_frozen


def test_linear_probe_features_example(tmp_path, tiny_checkpoint, image_dir):
    root, rows = image_dir
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(rows))
    npz = tmp_path / "split.npz"
    from surgfm_probe.cli import main

    main(["extract", "--checkpoint", str(tiny_checkpoint), "--manifest", str(manifest), "--root", str(root),
          "--label-field", "label.id", "--device", "cpu", "--num-workers", "0", "--out", str(npz)])  # fmt: skip
    stdout = _run("linear_probe_features.py", "--train", npz, "--val", npz, "--num-classes", 3,
                  "--output-dir", tmp_path / "run")  # fmt: skip
    assert "best_val_accuracy" in stdout and "predictions" in stdout
    assert np.isfinite(json.loads((tmp_path / "run" / "results.json").read_text())["best_val_accuracy"])
