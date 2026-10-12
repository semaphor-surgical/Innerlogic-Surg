"""Frozen-backbone feature extractor."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .backbone import feature_dim

_AUTOCAST = {None: None, "float32": None, "fp32": None, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


class FeatureExtractor(nn.Module):
    """Pooled image descriptor from a frozen ViT.

    The descriptor is `concat[CLS token of each of the last n_last_blocks blocks,
    mean patch token of the last block]`, with the final LayerNorm applied to every token.
    Its size is `D * (n_last_blocks + 1)` (2 * D by default); `use_avgpool=False` drops the
    mean-patch-token part.

    Keep parameters in float32. For lower precision pass `autocast_dtype="bfloat16"`;
    float16 is rejected because it overflows in this backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_last_blocks: int = 1,
        use_avgpool: bool = True,
        autocast_dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        super().__init__()
        if isinstance(autocast_dtype, torch.dtype):
            autocast_dtype = str(autocast_dtype).replace("torch.", "")
        if autocast_dtype not in _AUTOCAST:
            raise ValueError(
                f"autocast_dtype must be one of {[k for k in _AUTOCAST if k]} or None "
                f"(float16 produces NaNs with this backbone); got {autocast_dtype!r}"
            )
        self.backbone = backbone.eval().requires_grad_(False)
        self.n_last_blocks = n_last_blocks
        self.use_avgpool = use_avgpool
        self._autocast = _AUTOCAST[autocast_dtype]

    @property
    def feature_dim(self) -> int:
        return feature_dim(self.backbone, self.n_last_blocks, self.use_avgpool)

    def _autocast_ctx(self, device_type: str):
        return torch.autocast(device_type=device_type, dtype=self._autocast, enabled=self._autocast is not None)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """(B, 3, H, W) normalized images -> (B, feature_dim) float32 descriptors."""
        with self._autocast_ctx(images.device.type):
            layers = self.backbone.get_intermediate_layers(
                images, n=self.n_last_blocks, return_class_token=True
            )
        feats = torch.cat([cls for _, cls in layers], dim=-1)
        if self.use_avgpool:
            feats = torch.cat((feats, layers[-1][0].mean(dim=1)), dim=-1)
        return feats.float()

    @torch.no_grad()
    def dense(
        self,
        images: Tensor,
        upsampler: Optional[nn.Module] = None,
        output_size: Optional[Tuple[int, int]] = None,
        q_chunk_size: Optional[int] = None,
    ) -> Tensor:
        """(B, 3, H, W) -> (B, D, H/patch, W/patch) last-block patch features (float32).

        With an `upsampler` (see `load_anyup`) the map is upsampled to `output_size`, by default the
        input resolution, guided by the image; `q_chunk_size` trades speed for memory.
        """
        with self._autocast_ctx(images.device.type):
            (patches,) = self.backbone.get_intermediate_layers(images, n=1, reshape=True)
        patches = patches.float()
        if upsampler is None:
            return patches
        return upsampler(images.float(), patches, output_size=output_size, q_chunk_size=q_chunk_size)
