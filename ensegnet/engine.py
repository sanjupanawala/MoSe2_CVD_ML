"""Training, checkpoint inference and the training-fitted RGB reference."""
import gc
import json
import os
import platform
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import Tiles, augment, sha256
from .metrics import aggregate, confusion, overlap, tile_metrics
from .models import MEMBERS, load_model, make_model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def mixed_precision(device, enabled):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def region_loss(logits, y, ce_only=False):
    logits = logits.float()
    ce = F.cross_entropy(logits, y)
    if ce_only:
        return ce
    p = logits.softmax(1)
    target = F.one_hot(y, 3).permute(0, 3, 1, 2).float()
    intersection = (p * target).sum((0, 2, 3))
    denominator = (p + target).sum((0, 2, 3))
    return ce + 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def _settings(device, deterministic=False, training=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = training and not deterministic
    torch.backends.cudnn.allow_tf32 = True


@torch.inference_mode()
def validate(model, loader, device, amp):
    cm = np.zeros((3, 3), np.int64)
    model.eval()
    for _, x, y in loader:
        with mixed_precision(device, amp):
            logits = model(x.to(device))["logits"]
        p = logits.float().argmax(1).cpu().numpy()
        for truth, prediction in zip(y.numpy(), p):
            cm += confusion(truth, prediction)
    return {**overlap(cm), "confusion": cm.tolist()}


def train(root, config, output, model_kind="ensemble", device="cuda", asset=None, initialize=True):
    device = torch.device(device)
    source_leger = config["protocol"] == "leger-source"
    source_ensemble = config["protocol"] == "exfoliated"
    if source_leger and model_kind != "leger":
        raise ValueError("The leger-source protocol only applies to --model leger")
    if source_ensemble and model_kind not in (*MEMBERS, "ensemble"):
        raise ValueError("The exfoliated protocol is for the separately trained ensemble members")
    _settings(device, source_leger, training=True)
    kinds = MEMBERS if model_kind == "ensemble" else (model_kind,)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.pth")) or (output / "protocol.json").exists():
        raise FileExistsError("Choose a new output directory; existing runs are preserved")
    # Deliberately load only train/val here. Test data are read by evaluate().
    training = Tiles(root, "train", config["domain"], training=config["augmentation"] and not source_ensemble)
    validation = Tiles(root, "val", config["domain"])
    for split, data in [("train", training), ("val", validation)]:
        if len(data) != config["counts"][split]:
            raise ValueError(f"Unexpected {split} count: {len(data)}")
    protocol = {**config, "models": list(kinds), "pretrained_initialization": initialize,
                "device": str(device), "python": platform.python_version(), "torch": torch.__version__}
    write_json(output / "protocol.json", protocol)
    write_json(output / "training_manifest.json", {"train": training.records, "val": validation.records})
    protocol_sha = sha256(output / "protocol.json")
    for kind in kinds:
        for seed in config["seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = make_model(kind, pretrained=initialize, asset=asset, source_gray=source_leger).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
            # The archived source Leger run uses the global RNG. CVD uses an
            # explicit loader generator; source ensemble uses torch.randperm.
            generator = None if source_leger else torch.Generator().manual_seed(seed)
            loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True,
                                num_workers=0, generator=generator)
            val_loader = DataLoader(validation, batch_size=config["validation_batch_size"],
                                    generator=None if source_leger else torch.Generator().manual_seed(0))
            if source_ensemble:
                train_x = torch.stack([x.float() / 255 for _, x, _ in training.rows]).to(device)
                train_y = torch.stack([y for _, _, y in training.rows]).to(device)
            history, best, updates, skipped = [], -1., 0, 0
            path = output / f"{kind}__{seed}.pth"
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                total = 0.
                if source_ensemble:
                    order = torch.randperm(len(training))
                    def batches():
                        for start in range(0, len(order), config["batch_size"]):
                            indices = order[start:start + config["batch_size"]].to(device)
                            x, y = train_x[indices], train_y[indices]
                            pairs = [augment(a, b) for a, b in zip(x, y)]
                            yield None, torch.stack([a for a, _ in pairs]), torch.stack([b for _, b in pairs])
                    iterator = batches()
                else:
                    iterator = loader
                for _, x, y in iterator:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with mixed_precision(device, config["amp"]):
                        loss = region_loss(model(x)["logits"], y, ce_only=source_leger)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    scaler.scale(loss).backward()
                    scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped += int(scaler.get_scale() < scale)
                    updates += 1
                    total += float(loss.detach()) * len(x)
                score = validate(model, val_loader, device, config["amp"])
                history.append({"epoch": epoch, "train_loss": total / len(training), "validation": score})
                if score["miou"] > best:
                    best, best_epoch = score["miou"], epoch
                    torch.save({"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                "id": kind, "seed": seed, "epoch": epoch, "protocol_sha256": protocol_sha}, path)
                write_json(path.with_suffix(".json"), {"best_epoch": best_epoch, "best_validation_miou": best,
                           "updates": updates, "amp_skipped_updates": skipped, "epochs": history,
                           "checkpoint_sha256": sha256(path), "protocol_sha256": protocol_sha})
                print(f"{kind} fit {seed}: epoch {epoch}/{config['epochs']}, validation mIoU {score['miou']:.6f}", flush=True)
            del model, optimizer, scaler
            if source_ensemble:
                del train_x, train_y
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate(root, domain, output, checkpoints, kinds, device="cuda", amp=True,
             source_gray=False, split="test", reduction="cvd", batch_size=1):
    """One checkpoint or an equal-probability ensemble, scored once.

    For comparator three-fit means, evaluate each fit separately then summarize.
    A three-checkpoint probability average is a different experiment.
    """
    if len(checkpoints) != len(kinds) or not checkpoints:
        raise ValueError("Provide one model kind for every checkpoint")
    device = torch.device(device)
    _settings(device, source_gray)
    if reduction == "source":
        torch.backends.cuda.matmul.allow_tf32 = True
    data = Tiles(root, split, domain)
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Choose a new evaluation folder: {out}")
    (out / "predictions").mkdir(parents=True)
    # Load one network at a time to keep the nine-member GPU memory bounded.
    probability = torch.zeros((len(data), 3, *data.rows[0][2].shape), dtype=torch.float32, device=device)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(0))
    for checkpoint, kind in zip(checkpoints, kinds):
        model = load_model(checkpoint, kind, device, source_gray)
        offset = 0
        for _, x, _ in loader:
            with mixed_precision(device, amp):
                logits = model(x.to(device))["logits"]
            p = logits.float().softmax(1)
            # The two archived experiments used these respective reduction orders.
            probability[offset:offset + len(p)] += p / len(kinds) if reduction == "source" else p
            offset += len(p)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if reduction != "source":
        probability /= len(kinds)
    predictions = probability.argmax(1).cpu().numpy().astype(np.uint8)
    rows = []
    for (name, _, truth), prediction in zip(data.rows, predictions):
        rows.append(tile_metrics(name, truth.numpy(), prediction))
        Image.fromarray(prediction).save(out / "predictions" / (name + ".png"))
    result = aggregate(rows)
    result["metadata"] = {"domain": domain, "split": split, "class_order": ["substrate", "monolayer", "multilayer"],
                          "amp": amp and device.type == "cuda", "source_gray_255": source_gray,
                          "batch_size": batch_size, "reduction": reduction, "member_count": len(kinds),
                          "checkpoints": [{"name": Path(p).name, "kind": k, "sha256": sha256(p)}
                                          for p, k in zip(checkpoints, kinds)]}
    write_json(out / "metrics.json", result)
    write_json(out / "test_manifest.json", data.records)
    return result


def rgb_predict(x, bins, lut):
    q = np.minimum(x.astype(np.int32) * bins // 256, bins - 1)
    return lut[(q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]]


def fit_rgb(root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    training, validation = Tiles(root, "train"), Tiles(root, "val")
    best, search = None, []
    for bins in [8, 16, 32, 64]:
        hist = np.ones((bins ** 3, 3), np.float64)
        for _, x, y in training.rows:
            q = x.permute(1, 2, 0).numpy().astype(np.int32) * bins // 256
            ix = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]
            hist += np.bincount((ix * 3 + y.numpy()).ravel(), minlength=bins ** 3 * 3).reshape(-1, 3)
        lut = hist.argmax(1).astype(np.uint8)
        cm = np.zeros((3, 3), np.int64)
        for _, x, y in validation.rows:
            cm += confusion(y.numpy(), rgb_predict(x.permute(1, 2, 0).numpy(), bins, lut))
        score = overlap(cm)["miou"]
        search.append({"bins": bins, "validation_miou": score})
        if best is None or score > best[0]:
            best = score, bins, lut
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, bins=best[1], lut=best[2])
    write_json(output.with_suffix(".json"), {"validation_search": search, "selected_bins": best[1]})


def evaluate_rgb(root, checkpoint, output):
    data = Tiles(root, "test")
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    (out / "predictions").mkdir(parents=True)
    with np.load(checkpoint) as saved:
        bins, lut = int(saved["bins"]), saved["lut"]
    rows = []
    for name, x, y in data.rows:
        p = rgb_predict(x.permute(1, 2, 0).numpy(), bins, lut)
        rows.append(tile_metrics(name, y.numpy(), p))
        Image.fromarray(p).save(out / "predictions" / (name + ".png"))
    result = aggregate(rows)
    result["metadata"] = {"selected_bins": bins, "checkpoint_sha256": sha256(checkpoint)}
    write_json(out / "metrics.json", result)
    write_json(out / "test_manifest.json", data.records)
    return result


@torch.inference_mode()
def evaluate_transfer(root, checkpoint, output, normalized_images=None, full_field=False, device="cuda"):
    """Frozen source Leger checkpoint on CVD test images, in full precision.

    Primary normalization uses the archived upstream MATLAB-normalized images.
    Supply those exact PNGs explicitly; do not silently substitute a different
    Lab conversion. Without that folder this is the raw-patch sensitivity only.
    """
    device = torch.device(device)
    _settings(device, deterministic=True)
    torch.backends.cudnn.deterministic = False
    model = load_model(checkpoint, "leger", device, source_gray=True)
    data = Tiles(root, "test")
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    if full_field and normalized_images is None:
        raise ValueError("The recorded full-field sensitivity uses normalized images")
    (out / "predictions").mkdir(parents=True)
    rows, normalized_manifest = [], []
    for name, x, y in data.rows:
        if normalized_images is not None:
            path = Path(normalized_images) / (name + ".png")
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"))
            if rgb.shape != (512, 512, 3):
                raise ValueError(f"Normalized image has wrong size: {name}")
            normalized_manifest.append({"name": path.name, "sha256": sha256(path)})
        else:
            rgb = x.permute(1, 2, 0).numpy()
        height, width = rgb.shape[:2]
        if full_field:
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].float().to(device) / 255
            p = model(tensor)["logits"].argmax(1)[0].cpu().numpy().astype(np.uint8)
        else:
            padded = np.pad(rgb, ((0, (-height) % 96), (0, (-width) % 96), (0, 0)), mode="reflect")
            starts = [(r, c) for r in range(0, padded.shape[0], 96) for c in range(0, padded.shape[1], 96)]
            patches = np.stack([padded[r:r+96, c:c+96] for r, c in starts]).transpose(0, 3, 1, 2)
            predictions = []
            for start in range(0, len(patches), 4):
                tensor = torch.from_numpy(patches[start:start+4].copy()).float().to(device) / 255
                predictions.extend(model(tensor)["logits"].argmax(1).cpu().numpy())
            p = np.zeros(padded.shape[:2], np.uint8)
            for (r, c), patch in zip(starts, predictions):
                p[r:r+96, c:c+96] = patch
            p = p[:height, :width]
        rows.append(tile_metrics(name, y.numpy(), p))
        Image.fromarray(p).save(out / "predictions" / (name + ".png"))
    result = aggregate(rows)
    result["metadata"] = {"checkpoint_sha256": sha256(checkpoint), "source_training_only": True,
                          "mode": "normalized_full" if full_field else
                          ("normalized_patches" if normalized_images else "raw_patches"),
                          "normalized_inputs": normalized_manifest}
    write_json(out / "metrics.json", result)
    write_json(out / "test_manifest.json", data.records)
    return result
