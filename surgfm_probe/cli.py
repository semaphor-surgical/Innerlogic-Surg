"""Command line interface: ``surgfm-probe {extract,probe,tasks}``."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from . import __version__
from .backbone import ARCHS, load_backbone
from .data import ImageDataset, list_images, read_manifest
from .extract import extract_dataset, extract_video, load_features, save_features
from .features import FeatureExtractor
from .probe import DEFAULT_LR_GRID, probe_features, probe_standard
from .tasks import TASKS, format_tasks, get_task
from .transforms import eval_transform, full_frame_transform, train_transform

log = logging.getLogger("surgfm_probe")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--checkpoint", required=True, help="backbone .pth file (DINOv3 format)")
    g.add_argument("--arch", choices=sorted(ARCHS), help="override the architecture inferred from the checkpoint")
    g.add_argument("--n-last-blocks", type=int, default=1, help="CLS tokens of the last N blocks (default 1)")
    g.add_argument("--no-avgpool", action="store_true", help="drop the mean-patch-token half of the descriptor")
    g.add_argument("--autocast", choices=["float32", "bfloat16"], default="float32",
                   help="inference precision (float16 is not supported)")  # fmt: skip
    g.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu (default: cuda if available)")


def _add_image_args(p: argparse.ArgumentParser, batch_size: int = 128) -> None:
    g = p.add_argument_group("preprocessing")
    g.add_argument("--resize-size", type=int, default=256, help="short-side resize before the center crop")
    g.add_argument("--crop-size", type=int, default=224, help="center crop size (the model input size)")
    g.add_argument("--full-frame", metavar="SIZE", default=None,
                   help="instead resize the whole frame to SIZE or H,W (no crop); both multiples of 16")  # fmt: skip
    g.add_argument("--batch-size", type=int, default=batch_size,
                   help="images per batch" + (" (also the head training batch size; learning rates scale with it)" if batch_size == 256 else ""))
    g.add_argument("--num-workers", type=int, default=4)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="surgfm-probe", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p = sub.add_parser("extract", help="write frozen-backbone features for images, a manifest or a video")
    _add_model_args(p)
    _add_image_args(p)
    src = p.add_argument_group("input (one of)")
    src.add_argument("--images", help="directory of images (searched recursively)")
    src.add_argument("--manifest", help="CSV / JSON / JSONL manifest listing images")
    src.add_argument("--video", help="video file")
    p.add_argument("--root", help="directory that relative manifest image paths are resolved against")
    p.add_argument("--label-field", help="manifest label field (dotted path, e.g. phase.id); stored as `labels`")
    p.add_argument("--multi-label", action="store_true")
    p.add_argument("--num-classes", type=int)
    p.add_argument("--fps", type=float, default=1.0, help="video sampling rate in frames per second")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out", required=True, help="output .npz")

    # probe
    p = sub.add_parser("probe", help="train and evaluate a linear probe")
    _add_image_args(p, batch_size=256)
    p.add_argument("--checkpoint", help="backbone .pth (image-based probing)")
    p.add_argument("--arch", choices=sorted(ARCHS))
    p.add_argument("--n-last-blocks", type=int, default=1)
    p.add_argument("--no-avgpool", action="store_true")
    p.add_argument("--autocast", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--device", default=None)
    d = p.add_argument_group("data (manifests)")
    d.add_argument("--train", help="train manifest")
    d.add_argument("--val", help="validation manifest")
    d.add_argument("--test", help="test manifest")
    d.add_argument("--root", help="directory that relative image paths are resolved against")
    d.add_argument("--label-field", help="label field, dotted path (e.g. phase.id, label.id)")
    f = p.add_argument_group("data (precomputed features from `extract`; cached protocol only)")
    f.add_argument("--train-features")
    f.add_argument("--val-features")
    f.add_argument("--test-features")
    t = p.add_argument_group("task")
    t.add_argument("--task", choices=sorted(TASKS), help="preset for num_classes / selection split (see `tasks`)")
    t.add_argument("--num-classes", type=int)
    t.add_argument("--multi-label", action="store_true", help="multi-label targets, BCE loss, macro-F1 selection")
    t.add_argument("--select-on-test", action="store_true",
                   help="pick the best head on the test split when no separate --val is given")  # fmt: skip
    o = p.add_argument_group("protocol")
    o.add_argument("--protocol", choices=["standard", "cached"], default="standard",
                   help="standard: augmented, backbone runs every step (reported numbers). cached: features "
                        "extracted once, fast, ~1-2 points lower")  # fmt: skip
    o.add_argument("--epochs", type=int, default=10)
    o.add_argument("--epoch-length", type=int, default=1250, help="iterations per epoch")
    o.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    o.add_argument("--lr-grid", help="comma-separated base learning rates (default: 13-point grid 1e-5..0.1)")
    o.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output-dir", required=True)

    sub.add_parser("tasks", help="list the benchmark task presets")
    return parser


def _transforms(args):
    if args.full_frame:
        parts = [int(v) for v in str(args.full_frame).split(",")]
        ev = full_frame_transform(parts[0] if len(parts) == 1 else tuple(parts[:2]))
        return ev, ev
    return eval_transform(args.resize_size, args.crop_size), train_transform(args.crop_size)


def _extractor(args) -> FeatureExtractor:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_backbone(args.checkpoint, arch=args.arch, device=device)
    return FeatureExtractor(
        backbone,
        n_last_blocks=args.n_last_blocks,
        use_avgpool=not args.no_avgpool,
        autocast_dtype=args.autocast,
    )


def _meta(args, extractor: FeatureExtractor) -> dict:
    return {
        "package_version": __version__,
        "checkpoint": Path(args.checkpoint).name,
        "feature_dim": extractor.feature_dim,
        "n_last_blocks": args.n_last_blocks,
        "use_avgpool": not args.no_avgpool,
        "autocast": args.autocast,
        "resize_size": None if args.full_frame else args.resize_size,
        "crop_size": None if args.full_frame else args.crop_size,
        "full_frame": args.full_frame,
    }


def _cmd_extract(args) -> None:
    given = [n for n in ("images", "manifest", "video") if getattr(args, n)]
    if len(given) != 1:
        raise SystemExit("extract: pass exactly one of --images, --manifest, --video")
    extractor = _extractor(args)
    transform, _ = _transforms(args)
    meta = _meta(args, extractor)

    if args.video:
        out = extract_video(
            extractor, args.video, transform, fps=args.fps, max_frames=args.max_frames,
            batch_size=args.batch_size, device=args.device,
        )  # fmt: skip
        meta.update(source="video", fps=args.fps)
        save_features(args.out, out.pop("features"), meta, **out)
        log.info("wrote %s", args.out)
        return

    if args.images:
        root = Path(args.images)
        samples = list_images(root)
        dataset = ImageDataset(samples, transform=transform)
        paths = [str(p.relative_to(root)) for p in samples]
        labels_wanted = False
    else:
        samples = read_manifest(args.manifest)
        dataset = ImageDataset(
            samples, args.root, transform, args.label_field, args.multi_label, args.num_classes
        )
        paths = [str(dataset.image_path(i)) for i in range(len(dataset))]
        labels_wanted = bool(args.label_field)
    if not len(dataset):
        raise SystemExit("extract: no images found")
    log.info("extracting %d images, feature dim %d", len(dataset), extractor.feature_dim)
    features, labels = extract_dataset(
        extractor, dataset, batch_size=args.batch_size, num_workers=args.num_workers, device=args.device
    )
    meta.update(source="images", n=len(dataset))
    save_features(args.out, features, meta, paths=np.array(paths), labels=labels if labels_wanted else None)
    log.info("wrote %s  (%s)", args.out, features.shape)


def _same_file(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b and Path(a).resolve() == Path(b).resolve())


def _cmd_probe(args) -> None:
    task = get_task(args.task) if args.task else None
    multi_label = args.multi_label or (task.multi_label if task else False)
    num_classes = args.num_classes or (task.num_classes if task else None)
    lr_grid = tuple(float(v) for v in args.lr_grid.split(",")) if args.lr_grid else DEFAULT_LR_GRID

    use_features = bool(args.train_features)
    if use_features == bool(args.train):
        raise SystemExit("probe: pass either --train/--val/--test (manifests) or --train-features/... (not both)")
    if use_features and args.protocol == "standard":
        log.info("precomputed features given: using the cached protocol")
        args.protocol = "cached"
    if args.protocol == "standard" and not args.checkpoint:
        raise SystemExit("probe: --checkpoint is required unless precomputed features are used")

    train_ref, val_ref, test_ref = (
        (args.train_features, args.val_features, args.test_features)
        if use_features
        else (args.train, args.val, args.test)
    )
    if not test_ref and not val_ref:
        raise SystemExit("probe: need at least a validation or a test split")
    select_on_test = args.select_on_test
    if _same_file(val_ref, test_ref):
        log.warning("validation and test are the same file: the best head is selected on the test split")
        val_ref, select_on_test = None, True
    if task and task.selection_split == "test" and not select_on_test and not val_ref:
        select_on_test = True
    if not val_ref and not select_on_test:
        raise SystemExit(
            "probe: no --val given, so the best head would be chosen on the test split; add --select-on-test"
        )

    if use_features:
        def load(path):
            if not path:
                return None
            z = load_features(path)
            if "labels" not in z:
                raise SystemExit(f"{path}: no labels stored (extract with --manifest and --label-field)")
            return z["features"], z["labels"]

        train, val, test = load(train_ref), load(val_ref), load(test_ref)
        if num_classes is None:
            if multi_label:
                num_classes = train[1].shape[1]
            else:
                num_classes = int(max(s[1].max() for s in (train, val, test) if s is not None)) + 1
            log.info("num_classes inferred from the labels: %d", num_classes)
        result = probe_features(
            train, val, test, num_classes=num_classes, multi_label=multi_label, lr_grid=lr_grid,
            epochs=args.epochs, epoch_length=args.epoch_length, batch_size=args.batch_size,
            optimizer=args.optimizer, seed=args.seed, device=args.device,
            output_dir=args.output_dir, select_on_test=select_on_test,
        )  # fmt: skip
    else:
        if not args.label_field:
            raise SystemExit("probe: --label-field is required with manifests")
        if num_classes is None:
            raise SystemExit("probe: pass --num-classes or --task")
        eval_t, train_t = _transforms(args)

        def dataset(path, transform):
            if not path:
                return None
            return ImageDataset(
                read_manifest(path), args.root, transform, args.label_field, multi_label, num_classes
            )

        train_ds = dataset(train_ref, train_t if args.protocol == "standard" else eval_t)
        val_ds, test_ds = dataset(val_ref, eval_t), dataset(test_ref, eval_t)
        extractor = _extractor(args)
        if args.protocol == "standard":
            result = probe_standard(
                extractor, train_ds, val_ds, test_ds, num_classes=num_classes, multi_label=multi_label,
                lr_grid=lr_grid, epochs=args.epochs, epoch_length=args.epoch_length,
                batch_size=args.batch_size, optimizer=args.optimizer, seed=args.seed,
                num_workers=args.num_workers, device=args.device, output_dir=args.output_dir,
                select_on_test=select_on_test,
            )  # fmt: skip
        else:
            def feats(ds):
                if ds is None:
                    return None
                return extract_dataset(
                    extractor, ds, batch_size=args.batch_size, num_workers=args.num_workers, device=args.device
                )

            train, val, test = feats(train_ds), feats(val_ds), feats(test_ds)
            result = probe_features(
                train, val, test, num_classes=num_classes, multi_label=multi_label, lr_grid=lr_grid,
                epochs=args.epochs, epoch_length=args.epoch_length, batch_size=args.batch_size,
                optimizer=args.optimizer, seed=args.seed, device=args.device,
                output_dir=args.output_dir, select_on_test=select_on_test,
            )  # fmt: skip

    r = result.results
    (Path(args.output_dir) / "run_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    name = "macro-F1" if multi_label else "accuracy"
    print(f"\nprotocol {r['protocol']}   head selected on {r['selection_split']}   best head {r['best_classifier']}")
    print(f"val  {name}: {r['best_val_accuracy']:.2f}")
    if r["test_accuracy"] is not None:
        print(f"test {name}: {r['test_accuracy']:.2f}   (f1 macro {r['test_f1_macro']:.2f})")
    if r["selection_split"] == "test":
        print("note: the head was chosen on the test split itself, so this score is optimistic")
    print(f"results written to {Path(args.output_dir) / 'results.json'}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "tasks":
        print(format_tasks())
    elif args.command == "extract":
        _cmd_extract(args)
    else:
        _cmd_probe(args)


if __name__ == "__main__":
    main(sys.argv[1:])
