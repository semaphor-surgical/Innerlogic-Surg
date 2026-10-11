# surgfm-probe

Feature extraction and linear probing for the InnerLogic surgical vision foundation models
(ViT-B/16 and ViT-L/16 backbones in DINOv3 format).

- **Extract features** from image folders, manifests or videos with one command or three lines of Python.
- **Get dense patch features** for your own segmentation, tracking or retrieval pipelines.
- **Run a linear probe** with the same protocol that produced the reported frame-level classification scores.
- **Standalone**: inference-only model code is vendored, so the only requirements are PyTorch, torchvision, NumPy and Pillow.

## Contents

1. [Install](#install)
2. [Checkpoints](#checkpoints)
3. [Feature extraction](#feature-extraction)
4. [Linear probing](#linear-probing)
5. [Reproducing the reported results](#reproducing-the-reported-results)
6. [Verification](#verification)
7. [Scope](#scope)
8. [License](#license)

## Install

```bash
git clone <this repository>
cd surg_fm_probe
pip install -e .                  # torch, torchvision, numpy, pillow
pip install -e ".[video,progress]" # optional: video decoding (OpenCV) and progress bars
pip install -e ".[test]"          # optional: test dependencies
```

Python 3.9+ and PyTorch 2.1+. A GPU is recommended; everything also runs on CPU.

## Checkpoints

| Model | File | Embedding dim | Descriptor dim (default) |
|---|---|---|---|
| ViT-B/16 | `dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth` | 768 | 1536 |
| ViT-L/16 | `dinov3_vitl16_pretrain_innerlogic_surg-22622135.pth` | 1024 | 2048 |

The 8-hex suffix of each file name is the start of its SHA-256 (`sha256sum <file> | cut -c1-8`).
Download links: *to be added at release.*

A checkpoint is a flat backbone `state_dict`. The architecture (size, number of heads, FFN type) is
inferred from the tensors, so `--arch` is normally not needed. Wrapped checkpoints
(`{"model": ...}`, `{"teacher": ...}`) are unwrapped automatically.

## Feature extraction

### Python

```python
import torch
from PIL import Image
from surgfm_probe import FeatureExtractor, eval_transform, load_backbone

backbone = load_backbone("dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth", device="cuda")
extractor = FeatureExtractor(backbone)            # frozen, eval mode
transform = eval_transform()                      # resize 256 (bicubic) -> center crop 224 -> normalize

image = transform(Image.open("frame.png").convert("RGB")).unsqueeze(0).cuda()
descriptor = extractor(image)                     # (1, 1536) float32
dense = extractor.dense(image)                    # (1, 768, 14, 14) patch features, last block
```

A folder or manifest in one call:

```python
from surgfm_probe import ImageDataset, extract_dataset, list_images, save_features

dataset = ImageDataset(list_images("frames/"), transform=transform)
features, _ = extract_dataset(extractor, dataset, batch_size=128, num_workers=8)
save_features("frames.npz", features, paths=[str(p) for p in dataset.samples])
```

Per-frame features for a video (needs the `video` extra):

```python
from surgfm_probe import extract_video
out = extract_video(extractor, "case.mp4", transform, fps=1.0)
out["features"], out["frame_index"], out["time_s"]
```

### Command line

```bash
surgfm-probe extract --checkpoint vitb.pth --images frames/        --out frames.npz
surgfm-probe extract --checkpoint vitb.pth --manifest val.jsonl --root data/ --label-field phase.id --out val.npz
surgfm-probe extract --checkpoint vitb.pth --video case.mp4 --fps 1 --out case.npz
```

`python -m surgfm_probe ...` is equivalent. The output `.npz` holds `features` (N x D, float32), a JSON `meta` string
describing the settings, and `labels` / `paths` / `frame_index` / `time_s` where applicable. Read it back with
`surgfm_probe.load_features`.

### What the descriptor is

`descriptor = concat[ CLS token of the last n blocks, mean patch token of the last block ]`, with the final
LayerNorm applied to every token. With the defaults (`n = 1`) it has `2 x D` dimensions. Options:

| Flag | Effect |
|---|---|
| `--n-last-blocks N` | CLS tokens of the last N blocks (descriptor size `D x (N + 1)`) |
| `--no-avgpool` | drop the mean-patch-token half |
| `--autocast bfloat16` | bf16 inference (about 1e-4 cosine difference to fp32); fp16 is rejected because this backbone overflows in it |
| `--full-frame 224` / `--full-frame 224,384` | resize the whole frame (no crop); sides must be multiples of 16 |

### Preprocessing

The reported scores use `eval_transform(256, 224)`: bicubic resize of the short side to 256, center crop to 224,
ImageNet mean/std normalization. The backbone uses rotary position embeddings, so non-square and larger inputs
work (`--full-frame`), but scores were calibrated for the default.

## Linear probing

The probe trains a bank of linear heads on frozen descriptors, one per learning rate, and keeps the best one.

### Data format

A manifest lists images and labels. Three formats are accepted:

```jsonc
// JSON (list) or JSONL (one object per line); image paths are relative to --root
{"image": "case01/000123.png", "phase": {"id": 3}}
```

```csv
image,label
case01/000123.png,3
```

The label is read with a dotted path (`--label-field phase.id`; for CSV `--label-field label`). Class ids are
integers `0..C-1`. Multi-label targets (`--multi-label --num-classes C`) are lists of active class ids in JSON or
`;`-separated ids in CSV, are trained with BCE loss, and are scored with macro-F1.

### Run a probe

```bash
surgfm-probe probe \
    --checkpoint dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth \
    --train train.jsonl --val val.jsonl --test test.jsonl \
    --root /path/to/images --label-field phase.id --num-classes 7 \
    --output-dir runs/phase
```

Or with a benchmark preset (class count and selection split are filled in; see [Tasks](#tasks)):

```bash
surgfm-probe probe --task hyperkvasir --checkpoint vitb.pth \
    --train train.jsonl --val val.jsonl --test test.jsonl --root /path/to/HyperKvasir \
    --label-field label.id --output-dir runs/hyperkvasir
```

Python equivalent:

```python
from surgfm_probe import (FeatureExtractor, ImageDataset, eval_transform, load_backbone,
                          probe_standard, read_manifest, train_transform)

extractor = FeatureExtractor(load_backbone("vitb.pth", device="cuda"))
make = lambda f, tf: ImageDataset(read_manifest(f), root="/path/to/images", transform=tf, label_field="phase.id")
result = probe_standard(extractor, make("train.jsonl", train_transform()),
                        make("val.jsonl", eval_transform()), make("test.jsonl", eval_transform()),
                        num_classes=7, output_dir="runs/phase")
print(result.results["test_accuracy"], result.results["best_classifier"])
```

### Protocol

| | |
|---|---|
| Heads | 13 linear heads, learning rates 1e-5 ... 0.1 (`--lr-grid` to change), all trained at once on the same batches |
| Head init | weights N(0, 0.01), bias 0 |
| Optimizer | SGD, momentum 0.9, no weight decay (`--optimizer adamw` available), cosine schedule |
| Learning-rate scaling | `lr x batch_size / 256` |
| Schedule | 10 epochs x 1250 iterations, batch size 256 |
| Selection | after every epoch, the (epoch, head) pair with the best validation score is kept; test is evaluated with it |
| Metric | accuracy (single-label) or macro-F1 (multi-label); macro/weighted F1 are always reported too |
| Seed | `--seed 1234` |

Two protocols are available:

- **`--protocol standard`** (default): the backbone runs at every step on randomly resized crops with horizontal
  flips. This is the protocol behind the reported scores. It needs a GPU and is bounded by image loading
  (use `--num-workers`). Validation and test features are extracted once because they do not change.
- **`--protocol cached`**: features are extracted once with the evaluation transform and the heads are trained on
  them. It is much faster (minutes) and can start from `surgfm-probe extract` outputs
  (`--train-features train.npz --val-features val.npz --test-features test.npz`), but its scores differ from
  `standard` by up to a couple of points per task (lower on average). Do not mix the two protocols in one table.

### Outputs

`--output-dir` receives `results.json`, `best_head.pt` (the selected linear head) and `run_args.json`.
`results.json` includes `best_val_accuracy`, `test_accuracy`, `test_f1_macro`, `test_f1_weighted`, `best_classifier`,
per-epoch metrics, the protocol, the selection split, and the sizes of the splits. When a task has multi-label
targets, `primary_metric` says that the field named `*_accuracy` holds macro-F1.

### Tasks

`surgfm-probe tasks` lists the presets for the eleven frame-level classification benchmarks:

| `--task` | Benchmark | Classes | Head selected on |
|---|---|---|---|
| `cholec80-phase` | Cholec80 phase | 7 | test |
| `autolaparo-phase` | AutoLaparo phase | 7 | val |
| `multibypass-phase` | MultiBypass140 phase | 12 | val |
| `multibypass-step` | MultiBypass140 step | 46 | val |
| `pitvis-step` | PitVis step | 15 | val |
| `hyperkvasir` | HyperKvasir | 23 | val |
| `cholecinstanceseg` | CholecInstanceSeg instrument | 7 | test |
| `heichole` | HeiChole instrument | 7 | test |
| `sarrarp50` | SAR-RARP50 gesture | 7 | test |
| `surgvu` | SurgVU instrument | 20 | test |
| `lapgckvad` | LapGC-KVAD-30 anatomy | 15 | test |

Datasets and annotations are **not** distributed here; obtain them from their original sources and write one
manifest per split.

**Selection on test.** Benchmarks without a validation split have no held-out data to select a head on. For these
the head is selected on the same images that are reported, which makes the score optimistic. The CLI requires
`--select-on-test` (or a task preset that implies it), prints a warning, and records `selection_split` in
`results.json`. Prefer to carve out a validation split from your training data when you can.

## Reproducing the reported results

Test scores of the released checkpoints with `--protocol standard` (single run, seed 1234). `--` means the
configuration has not been evaluated yet.

| Model | ChPh | AuPh | MbPh | MbSt | PitV | HypK | ChIS | HeiC | SARr | SgVU | LapG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ViT-B/16 | 80.7 | 72.9 | 84.9 | 74.0 | 74.5 | 88.9 | 67.0 | 50.7 | 40.8 | 23.9 | 72.1 |
| ViT-L/16 | -- | -- | -- | -- | 74.5 | 90.0 | 72.7 | 56.7 | 43.0 | 26.9 | 73.7 |

ChPh, ChIS, HeiC, SARr, SgVU and LapG are selected on the test split (see above). Column names:
ChPh = Cholec80 phase, AuPh = AutoLaparo phase, MbPh/MbSt = MultiBypass140 phase/step, PitV = PitVis step,
HypK = HyperKvasir, ChIS = CholecInstanceSeg instrument, HeiC = HeiChole instrument, SARr = SAR-RARP50 gesture,
SgVU = SurgVU instrument, LapG = LapGC-KVAD-30 anatomy. The metric is accuracy for all of them.

Expect run-to-run differences of a few tenths of a point from augmentation randomness and GPU nondeterminism.

Reproduction check: running this package on HyperKvasir with the ViT-B checkpoint (`--task hyperkvasir`,
`--protocol standard`) gives 88.9 test / 89.9 validation accuracy, matching the 88.9 above. The `cached`
protocol gives 89.7 test on the same task.

## Verification

```bash
pytest                                                    # CPU tests, a few seconds

# bit-exact comparison with the reference DINOv3 implementation
python scripts/parity_check.py --dinov3-repo /path/to/dinov3 --checkpoint vitb.pth
```

The parity script loads the checkpoint into both the reference model and the vendored one with `strict=True` and
compares CLS tokens, patch tokens and the descriptor on square and non-square inputs. The maximum difference is 0
for both released checkpoints.

## Scope

This repository covers feature extraction and frame-level linear classification. Dense-prediction probes
(segmentation, depth), clip-level probes (critical view of safety, triplets) and video probes are not included.

## License

The model code in `surgfm_probe/models/` is derived from DINOv3 and, together with the weights, is subject to the
DINOv3 License; see [LICENSE.md](LICENSE.md) and [NOTICE.md](NOTICE.md). Read them before redistributing or using
the software or checkpoints commercially.
