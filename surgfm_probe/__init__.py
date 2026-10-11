"""Feature extraction and linear probing for DINOv3-format ViT backbones."""

__version__ = "0.1.0"

from .backbone import ARCHS, load_backbone
from .data import ImageDataset, list_images, read_manifest
from .extract import extract_dataset, extract_video, load_features, save_features
from .features import FeatureExtractor
from .probe import DEFAULT_LR_GRID, probe_features, probe_standard
from .transforms import eval_transform, full_frame_transform, train_transform

__all__ = [
    "ARCHS",
    "DEFAULT_LR_GRID",
    "FeatureExtractor",
    "ImageDataset",
    "eval_transform",
    "extract_dataset",
    "extract_video",
    "full_frame_transform",
    "load_backbone",
    "list_images",
    "load_features",
    "probe_features",
    "probe_standard",
    "read_manifest",
    "save_features",
    "train_transform",
]
