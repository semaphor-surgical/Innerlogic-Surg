# Provenance

`surgfm_probe/models/` is an inference-only re-implementation of the ViT backbone from Meta AI's **DINOv3**
(https://github.com/facebookresearch/dinov3). It keeps the official parameter and buffer names so that
DINOv3-format checkpoints load with `strict=True`; `scripts/parity_check.py` compares its outputs with the
official implementation. The code is governed by the **DINOv3 License** (see `LICENSE.md`), and the file headers
under `surgfm_probe/models/` keep the upstream notice.

The checkpoints this package is meant to load are in DINOv3 format and are subject to the same license.

The rest of the package (feature extraction, data handling, linear-probe protocol, CLI) is new code. It is
distributed together with the DINOv3-derived parts under the same license (see `LICENSE.md`).

`surgfm_probe.load_anyup` optionally loads the AnyUp feature upsampler (https://github.com/wimmerth/anyup, CC BY 4.0)
through `torch.hub`. AnyUp's code and weights are not part of this repository and are not covered by the DINOv3
License; they are downloaded from their own repository at the user's request.
