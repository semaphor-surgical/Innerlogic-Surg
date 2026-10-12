"""Use the backbone as the image encoder of a larger model (here: a small segmentation model).

    python examples/use_as_encoder.py --checkpoint vitb.pth            # frozen encoder
    python examples/use_as_encoder.py --checkpoint vitb.pth --finetune  # encoder trained end to end
"""

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from surgfm_probe import load_backbone


class SurgEncoder(nn.Module):
    """(B, 3, H, W) ImageNet-normalized images -> CLS (B, D) and patch feature map (B, D, H/16, W/16)."""

    def __init__(self, checkpoint, freeze=True):
        super().__init__()
        self.backbone = load_backbone(checkpoint)  # float32, requires_grad=False
        self.dim = self.backbone.embed_dim
        self.freeze = freeze
        self.backbone.requires_grad_(not freeze)

    def forward(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            ((fmap, cls),) = self.backbone.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)
        return cls, fmap


class SegModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Conv2d(encoder.dim, num_classes, kernel_size=1)

    def forward(self, x):
        _, fmap = self.encoder(x)
        return F.interpolate(self.head(fmap), size=x.shape[-2:], mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--num-classes", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SurgEncoder(args.checkpoint, freeze=not args.finetune)
    model = SegModel(encoder, args.num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    images = torch.randn(2, 3, 224, 224, device=device)  # stand-in for normalized frames
    target = torch.randint(0, args.num_classes, (2, 224, 224), device=device)
    loss = F.cross_entropy(model(images), target)
    loss.backward()
    optimizer.step()
    print(f"trainable parameters: {sum(p.numel() for p in params):,}  loss: {loss.item():.3f}")


if __name__ == "__main__":
    main()
