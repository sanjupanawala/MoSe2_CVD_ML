"""Three-class pooled overlap, coverage error and boundary agreement."""
import numpy as np
import cv2


def confusion(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or y.ndim != 2 or y.size == 0:
        raise ValueError("Expected matching, nonempty 2-D class maps")
    if not set(np.unique(y)) <= {0, 1, 2} or not set(np.unique(p)) <= {0, 1, 2}:
        raise ValueError("Class maps must contain only 0, 1, 2")
    return np.bincount((3 * y.astype(np.int64) + p).ravel(), minlength=9).reshape(3, 3)


def overlap(cm):
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (3, 3) or (cm < 0).any() or cm.sum() == 0:
        raise ValueError("Expected a nonempty 3x3 confusion matrix")
    tp = cm.diagonal()
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros(3), where=union > 0)
    denom = cm.sum(0) + cm.sum(1)
    dice = np.divide(2 * tp, denom, out=np.zeros(3), where=denom > 0)
    return {"miou": float(iou.mean()), "iou": iou.tolist(), "dice": dice.tolist(),
            "pixel_accuracy": float(tp.sum() / cm.sum())}


def boundary_f1(y, p, c, tolerance=2):
    kernel = np.ones((3, 3), np.uint8)
    a = cv2.morphologyEx((y == c).astype("uint8"), cv2.MORPH_GRADIENT, kernel) > 0
    b = cv2.morphologyEx((p == c).astype("uint8"), cv2.MORPH_GRADIENT, kernel) > 0
    if not a.any() and not b.any():
        return None
    if not a.any() or not b.any():
        return 0.
    dilation = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    precision = (b & (cv2.dilate(a.astype("uint8"), dilation) > 0)).sum() / b.sum()
    recall = (a & (cv2.dilate(b.astype("uint8"), dilation) > 0)).sum() / a.sum()
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.


def tile_metrics(name, y, p):
    cm = confusion(y, p)
    truth = [float((y == c).mean() * 100) for c in range(3)]
    prediction = [float((p == c).mean() * 100) for c in range(3)]
    return {"name": name, "confusion": cm.tolist(), **overlap(cm),
            "true_coverage_percent": truth, "predicted_coverage_percent": prediction,
            "coverage_error_pp": [b - a for a, b in zip(truth, prediction)],
            "boundary_f1": [boundary_f1(y, p, c) for c in range(3)]}


def aggregate(rows):
    if not rows:
        raise ValueError("No predictions to score")
    cm = np.array([r["confusion"] for r in rows], dtype=np.int64).sum(0)
    boundary = []
    for c in range(3):
        values = [r["boundary_f1"][c] for r in rows if r["boundary_f1"][c] is not None]
        boundary.append(float(np.mean(values)) if values else None)
    return {**overlap(cm), "confusion": cm.tolist(), "n_tiles": len(rows),
            "coverage_mae_pp": np.abs([r["coverage_error_pp"] for r in rows]).mean(0).tolist(),
            "boundary_f1": boundary, "per_image": rows}
