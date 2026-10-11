"""Train a linear probe on precomputed descriptors (the fast `cached` protocol) and use the best head.

    surgfm-probe extract --checkpoint vitb.pth --manifest train.jsonl --root data --label-field phase.id --out train.npz
    surgfm-probe extract --checkpoint vitb.pth --manifest val.jsonl   --root data --label-field phase.id --out val.npz
    python examples/linear_probe_features.py --train train.npz --val val.npz --num-classes 7
"""

import argparse

import torch

from surgfm_probe import load_features, probe_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--output-dir", default="runs/example")
    args = ap.parse_args()

    train, val = load_features(args.train), load_features(args.val)
    test = load_features(args.test) if args.test else None
    as_pair = lambda d: (d["features"], d["labels"])

    result = probe_features(
        as_pair(train), as_pair(val), as_pair(test) if test else None,
        num_classes=args.num_classes, output_dir=args.output_dir,
    )  # fmt: skip
    print({k: result.results[k] for k in ("best_val_accuracy", "test_accuracy", "best_classifier") if k in result.results})

    head = result.head  # torch.nn.Linear
    with torch.no_grad():
        logits = head(torch.as_tensor(val["features"][:4]).to(next(head.parameters()).device))
    print("predictions for the first 4 validation frames:", logits.argmax(1).tolist())


if __name__ == "__main__":
    main()
