# Innerlogic-Surg

Surgical vision foundation models (ViT-B/16 and ViT-L/16, DINOv3 format) with tools to extract features,
run linear probes, and use the backbone as an encoder inside your own models. The only requirements are
PyTorch, torchvision, NumPy and Pillow.

## Install

```bash
git clone <this repository>
cd Innerlogic-Surg
pip install -e .                    # torch, torchvision, numpy, pillow
pip install -e ".[video,progress]"  # optional: video decoding (OpenCV), progress bars
```

Python 3.9+ and PyTorch 2.1+. A GPU is recommended; everything also runs on CPU.

## Checkpoints

| Model | File | Embedding dim | Descriptor dim |
|---|---|---|---|
| ViT-B/16 | `dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth` | 768 | 1536 |
| ViT-L/16 | `dinov3_vitl16_pretrain_innerlogic_surg-22622135.pth` | 1024 | 2048 |

The 8-hex suffix is the start of the file's SHA-256. Download links: *to be added at release.*
A checkpoint is a flat backbone `state_dict` with DINOv3 key names; the architecture is inferred from it.

## Extract features

```python
from PIL import Image
from surgfm_probe import FeatureExtractor, eval_transform, load_backbone

extractor = FeatureExtractor(load_backbone("dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth", device="cuda"))
transform = eval_transform()        # resize 256 (bicubic) -> center crop 224 -> ImageNet mean/std

x = transform(Image.open("frame.png").convert("RGB")).unsqueeze(0).cuda()
descriptor = extractor(x)           # (1, 1536) pooled image descriptor
dense = extractor.dense(x)          # (1, 768, 14, 14) patch feature map
```

Command line (folders, manifests or videos; writes an `.npz`, read back with `surgfm_probe.load_features`):

```bash
surgfm-probe extract --checkpoint vitb.pth --images frames/ --out frames.npz
surgfm-probe extract --checkpoint vitb.pth --manifest val.jsonl --root data/ --label-field phase.id --out val.npz
surgfm-probe extract --checkpoint vitb.pth --video case.mp4 --fps 1 --out case.npz     # needs the video extra
```

The descriptor is `concat[CLS token, mean patch token]` of the last block (`2 x D`). `--n-last-blocks N` adds the
CLS tokens of more blocks, `--no-avgpool` drops the mean-patch half, `--autocast bfloat16` runs in bf16
(fp16 overflows in this backbone), and `--full-frame 224,384` resizes whole frames without cropping
(sides must be multiples of 16). Scripts: `examples/extract_features.py`.

### Pixel-resolution dense features (AnyUp)

The patch map is coarse (`H/16 x W/16`). Pass an [AnyUp](https://github.com/wimmerth/anyup) upsampler to lift it to
pixel resolution, guided by the image:

```python
from surgfm_probe import load_anyup

upsampler = load_anyup(device="cuda")                        # public AnyUp weights, fetched on first use
dense_hr = extractor.dense(x, upsampler=upsampler)           # (1, 768, 224, 224)
dense_112 = extractor.dense(x, upsampler=upsampler, output_size=(112, 112), q_chunk_size=256)  # (1, 768, 112, 112), lower memory
```

AnyUp is a separate project (Wimmer et al., CC BY 4.0) that is neither bundled nor retrained here. `load_anyup`
downloads a pinned revision of its code and its pretrained weights through `torch.hub` (needs network access and
Python 3.10+; the repository is trusted automatically, pass `trust_repo=False` to be asked first).
`load_anyup("single_backbone")` selects the model from the AnyUp paper instead of the default multi-backbone one.
`examples/extract_features.py --anyup` shows it end to end. Please cite the AnyUp paper if you use it.

## Linear probe

Trains linear heads on frozen features of your own labeled frames and reports test accuracy.
Datasets are not distributed here; write one manifest per split (JSON, JSONL or CSV):

```jsonc
{"image": "case01/000123.png", "phase": {"id": 3}}      // image path is relative to --root
```

```bash
surgfm-probe probe --checkpoint vitb.pth \
    --train train.jsonl --val val.jsonl --test test.jsonl \
    --root /path/to/images --label-field phase.id --num-classes 7 --output-dir runs/phase
```

The class id is read with a dotted path (`--label-field`) and must be an integer in `0..C-1`. Multi-label tasks use
`--multi-label` (BCE loss, macro-F1). `--task NAME` fills in the class count and selection split for the 11 benchmark
presets listed by `surgfm-probe tasks`. Results go to `results.json` (accuracy, F1, per-epoch metrics),
`best_head.pt` and `run_args.json`. Python equivalent:

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

**Protocol.** 13 linear heads (one per learning rate, 1e-5 to 0.1) are trained at once with SGD (momentum 0.9,
cosine schedule, lr scaled by batch/256), 10 epochs of 1250 iterations, batch size 256, seed 1234. The
(epoch, head) pair with the best validation score is evaluated on test. `--protocol standard` (default) is the
setting used for the reported frame-level scores: random-resized-crop and flip augmentation with the backbone
running at every step. `--protocol cached` extracts features once and trains in minutes (also from
`surgfm-probe extract` outputs via `--train-features/--val-features/--test-features`; see
`examples/linear_probe_features.py`), but its scores differ from `standard` by up to a couple of points, so do
not mix the two in one table.

If a dataset has no validation split, `--select-on-test` selects the head on the test images, which makes
the score optimistic; the CLI warns and records `selection_split` in `results.json`.

## Use as an encoder in your own model

`load_backbone` returns a plain `torch.nn.Module` (float32, frozen, eval mode). It takes `(B, 3, H, W)`
ImageNet-normalized images with `H` and `W` multiples of 16 (any size or aspect ratio) and exposes:

```python
from surgfm_probe import FeatureExtractor, load_backbone

backbone = load_backbone("vitb.pth", device="cuda")      # backbone.embed_dim, .patch_size, .n_blocks = 768, 16, 12

((fmap, cls),) = backbone.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)
#   fmap: (B, D, H/16, W/16) patch feature map, cls: (B, D)  -> segmentation, detection, tracking, retrieval heads
(tokens,) = backbone.get_intermediate_layers(x, n=1)
#   tokens: (B, N, D) patch tokens                           -> transformer decoders, vision-language connectors
maps = backbone.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True)
#   4 feature maps from block indices 2, 5, 8, 11 (ViT-L has 24 blocks: use [5, 11, 17, 23])  -> UPerNet / DPT-style decoders
descriptor = FeatureExtractor(backbone)(x)               # (B, 2D) pooled descriptor, no gradients
hr = upsampler(x, fmap)                                  # optional AnyUp upsampler (see above): (B, D, H, W)
```

Wrap it as a submodule, frozen or fine-tuned end to end (the backbone has no dropout or BatchNorm, so
`train()` and `eval()` behave the same):

```python
import torch
from torch import nn
from surgfm_probe import load_backbone

class SurgEncoder(nn.Module):
    def __init__(self, checkpoint, freeze=True):
        super().__init__()
        self.backbone = load_backbone(checkpoint)
        self.dim, self.freeze = self.backbone.embed_dim, freeze
        self.backbone.requires_grad_(not freeze)

    def forward(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            ((fmap, cls),) = self.backbone.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)
        return cls, fmap

encoder = SurgEncoder("vitb.pth", freeze=True)           # freeze=False to fine-tune
head = nn.Conv2d(encoder.dim, num_classes, 1)            # any head on top
```

A complete runnable version with a segmentation head and one training step is in
`examples/use_as_encoder.py`. For reduced precision use `torch.autocast("cuda", dtype=torch.bfloat16)`
(not fp16). The same checkpoint file also loads with `strict=True` into the official DINOv3
`DinoVisionTransformer`.

## License

The model code in `surgfm_probe/models/` is derived from DINOv3 and, together with the weights, is subject to the
DINOv3 License; see [LICENSE.md](LICENSE.md) and [NOTICE.md](NOTICE.md). Read them before redistributing or using
the software or checkpoints commercially.
