"""Extract descriptors and dense patch features for the images in a folder.

    python examples/extract_features.py --checkpoint vitb.pth --images frames/ --out frames.npz
"""

import argparse

import torch

from surgfm_probe import (
    FeatureExtractor,
    ImageDataset,
    eval_transform,
    extract_dataset,
    list_images,
    load_anyup,
    load_backbone,
    save_features,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="features.npz")
    ap.add_argument("--anyup", action="store_true", help="also upsample the dense features with AnyUp (needs network)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor(load_backbone(args.checkpoint, device=device))
    paths = list_images(args.images)
    dataset = ImageDataset(paths, transform=eval_transform())

    features, _ = extract_dataset(extractor, dataset, batch_size=64, num_workers=4, device=device)
    save_features(args.out, features, paths=[str(p) for p in paths])
    print(f"{features.shape[0]} images -> {features.shape[1]}-d descriptors in {args.out}")

    image, _ = dataset[0]
    dense = extractor.dense(image.unsqueeze(0).to(device))
    print("dense patch features of the first image:", tuple(dense.shape))
    if args.anyup:
        sharp = extractor.dense(image.unsqueeze(0).to(device), upsampler=load_anyup(device=device))
        print("AnyUp-upsampled dense features:", tuple(sharp.shape))


if __name__ == "__main__":
    main()
