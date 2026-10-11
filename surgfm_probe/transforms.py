"""Image preprocessing. `eval_transform()` defaults are the ones used for the reported scores."""

from typing import Tuple, Union

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_NORMALIZE = [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]


def eval_transform(resize_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    """Resize the short side to `resize_size` (bicubic), then center-crop `crop_size` x `crop_size`."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            *_NORMALIZE,
        ]
    )


def full_frame_transform(size: Union[int, Tuple[int, int]]) -> transforms.Compose:
    """Resize the whole frame to `size` (an int or an (H, W) pair) without cropping.

    Both sides must be multiples of the patch size. RoPE lets the backbone take non-square
    input, but scores are only calibrated for `eval_transform()`.
    """
    if isinstance(size, int):
        size = (size, size)
    return transforms.Compose(
        [transforms.Resize(tuple(size), interpolation=InterpolationMode.BICUBIC), *_NORMALIZE]
    )


def train_transform(crop_size: int = 224) -> transforms.Compose:
    """Augmentation for the `standard` probe protocol: random resized crop + horizontal flip."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(crop_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            *_NORMALIZE,
        ]
    )
