"""Optional AnyUp feature upsampling (https://github.com/wimmerth/anyup, CC BY 4.0).

AnyUp is a separate project that is not bundled here: its code and weights are fetched through
`torch.hub` the first time `load_anyup` is called, so that call needs network access and Python 3.10+.
"""

import torch
from torch import nn

ANYUP_REPO = "wimmerth/anyup:351807a9c4287368732cc247f26c7c81c9139af4"  # pinned revision
_VARIANTS = {"multi_backbone": "anyup_multi_backbone", "single_backbone": "anyup"}


def load_anyup(
    variant: str = "multi_backbone", device="cpu", repo: str = ANYUP_REPO, trust_repo: bool = True, **hub_kwargs
) -> nn.Module:
    """Load the pretrained AnyUp upsampler as a frozen, eval-mode module.

    `variant="multi_backbone"` is trained on features of several encoders and is the better fit for an
    encoder it has not seen; `"single_backbone"` is the model of the AnyUp paper. Extra keyword arguments
    (e.g. `use_natten=True`) go to the AnyUp hub entry point.

    The code is fetched at a pinned commit and run, so `torch.hub`'s interactive "trust this repository?"
    question (which hangs or fails in scripts) is answered for you; pass `trust_repo=False` to be asked instead.

    Call it as `upsampler(image, features, output_size=None, q_chunk_size=None)`, where `image` is the
    ImageNet-normalized (B, 3, H, W) input of the encoder and `features` is a (B, C, h, w) feature map.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_VARIANTS)}; got {variant!r}")
    model = torch.hub.load(repo, _VARIANTS[variant], trust_repo=trust_repo, device=device, **hub_kwargs)
    return model.eval().requires_grad_(False)
