"""Small command-line interface; run python -m ensegnet --help."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="2DM-Ensegnet research workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train from a frozen JSON configuration; never read the test split")
    train.add_argument("--data", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--model", choices=["ensemble", "segformer_b1", "student_control", "scse", "han", "leger"], default="ensemble")
    train.add_argument("--device", default="cuda")
    train.add_argument("--asset", help="Local MiT-B1 directory or VGG16 initialization file")
    train.add_argument("--random-init", action="store_true", help="Diagnostic only; changes the published protocol")
    evaluate = sub.add_parser("evaluate", help="Evaluate a single output or equal-probability ensemble")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--domain", choices=["cvd", "exfoliated"], default="cvd")
    evaluate.add_argument("--weights-dir", help="Folder with the nine <architecture>__<seed>.pth files")
    evaluate.add_argument("--checkpoints", nargs="+")
    evaluate.add_argument("--models", nargs="+",
                          choices=["segformer_b1", "student_control", "scse", "han", "leger"])
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--full-precision", action="store_true")
    evaluate.add_argument("--source-gray", action="store_true", help="Leger source reproduction uses grayscale 0..255")
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--reduction", choices=["cvd", "source"], default="cvd")
    evaluate.add_argument("--batch-size", type=int, default=1)
    for command in ["fit-rgb", "evaluate-rgb"]:
        p = sub.add_parser(command)
        p.add_argument("--data", required=True)
        p.add_argument("--output", required=True)
        if command == "evaluate-rgb":
            p.add_argument("--checkpoint", required=True)
    summary = sub.add_parser("summarize", help="Arithmetic mean of separately evaluated comparator fits")
    summary.add_argument("metrics", nargs="+")
    summary.add_argument("--output", required=True)
    transfer = sub.add_parser("transfer", help="Frozen source Leger model on CVD test images")
    transfer.add_argument("--data", required=True)
    transfer.add_argument("--checkpoint", required=True)
    transfer.add_argument("--output", required=True)
    transfer.add_argument("--normalized-images", help="Exact archived MATLAB Lab-normalized PNG folder")
    transfer.add_argument("--full-field", action="store_true")
    transfer.add_argument("--device", default="cuda")
    plot = sub.add_parser("plot", help="Create comparison plots from saved metrics, without running a model")
    plot.add_argument("--metrics", nargs="+", required=True)
    plot.add_argument("--labels", nargs="+", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "plot":
        from .figures import plot
        plot(args.metrics, args.labels, args.output)
        return
    # Match the source scripts' cuBLAS workspace setting before CUDA starts.
    if args.command == "transfer" or (args.command == "evaluate" and args.domain == "exfoliated"):
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from . import engine
    if args.command == "train":
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if config["protocol"] in {"exfoliated", "leger-source"}:
            import os
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        engine.train(args.data, config, args.output, args.model, args.device, args.asset, not args.random_init)
    elif args.command == "evaluate":
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        if args.weights_dir:
            if args.models or args.checkpoints:
                parser.error("Use --weights-dir or explicit --models/--checkpoints, not both")
            from .models import MEMBERS
            args.models = [kind for kind in MEMBERS for _ in range(3)]
            args.checkpoints = [str(Path(args.weights_dir) / f"{kind}__{seed}.pth")
                                for kind in MEMBERS for seed in [20261010, 20261011, 20261012]]
        if not args.models or not args.checkpoints:
            parser.error("Provide --weights-dir or both --models and --checkpoints")
        if args.source_gray and (args.models != ["leger"] or not args.full_precision):
            parser.error("Use --source-gray with one Leger checkpoint and --full-precision")
        result = engine.evaluate(args.data, args.domain, args.output, args.checkpoints, args.models,
                                 args.device, not args.full_precision, args.source_gray, args.split,
                                 args.reduction, args.batch_size)
        print(f"mIoU: {result['miou']:.9f}; coverage MAE (pp): {result['coverage_mae_pp']}")
    elif args.command == "fit-rgb":
        if Path(args.output).suffix != ".npz":
            parser.error("Use an .npz output filename")
        engine.fit_rgb(args.data, args.output)
    elif args.command == "evaluate-rgb":
        result = engine.evaluate_rgb(args.data, args.checkpoint, args.output)
        print(f"RGB mIoU: {result['miou']:.9f}")
    elif args.command == "summarize":
        import numpy as np
        rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.metrics]
        identities = [[(r["name"], r["true_coverage_percent"]) for r in row["per_image"]] for row in rows]
        if any(identity != identities[0] for identity in identities[1:]):
            parser.error("Comparator fits must be scored on the same ordered reference images")
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        result = {key: np.mean([r[key] for r in rows], axis=0).tolist()
                  for key in ["miou", "iou", "dice", "pixel_accuracy", "coverage_mae_pp", "boundary_f1"]}
        result["metadata"] = {"aggregation": "arithmetic mean of separately scored fits", "n_fits": len(rows)}
        engine.write_json(args.output, result)
    elif args.command == "transfer":
        result = engine.evaluate_transfer(args.data, args.checkpoint, args.output, args.normalized_images,
                                          args.full_field, args.device)
        print(f"Frozen source-to-CVD mIoU: {result['miou']:.9f}")
