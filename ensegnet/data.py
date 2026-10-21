"""Explicit split loading; never create or reshuffle a dataset partition."""
import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def augment(x, y):
    """The archived per-image training transforms; labels stay registered."""
    n = random.randrange(4)
    x, y = torch.rot90(x, n, [-2, -1]), torch.rot90(y, n, [-2, -1])
    if random.random() < .5:
        x, y = x.flip(-1), y.flip(-1)
    if random.random() < .5:
        x, y = x.flip(-2), y.flip(-2)
    if random.random() < .75:
        x = (x * random.uniform(.85, 1.15) + random.uniform(-.08, .08)).clamp(0, 1)
    return x, y


class Tiles(Dataset):
    def __init__(self, root, split, domain="cvd", training=False):
        if split not in {"train", "val", "test"}:
            raise ValueError("Choose an existing train, val or test split")
        if training and split != "train":
            raise ValueError("Augmentation is only allowed on the training split")
        self.rows, self.records = [], []
        self.training = training
        folder = Path(root) / split
        image_paths = sorted(p for p in (folder / "images").iterdir()
                             if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
        if not image_paths:
            raise ValueError(f"No images in {folder / 'images'}")
        if len({p.stem for p in image_paths}) != len(image_paths):
            raise ValueError("Image basenames must be unique within a split")
        mask_dir = folder / ("masks" if domain == "cvd" else "labels")
        mask_paths = [p for p in mask_dir.iterdir() if p.suffix.lower() == ".png"]
        masks = {p.stem: p for p in mask_paths}
        expected_masks = {p.stem + ("_mask" if domain == "cvd" else "") for p in image_paths}
        if len(masks) != len(mask_paths) or set(masks) != expected_masks:
            raise ValueError(f"Image/mask pairing differs in {folder}")
        for path in image_paths:
            mask = masks[path.stem + ("_mask" if domain == "cvd" else "")]
            with Image.open(path) as image:
                x = np.array(image.convert("RGB"))
            with Image.open(mask) as image:
                y = np.array(image)
            expected = (512, 512) if domain == "cvd" else (100, 100)
            if x.shape != (*expected, 3) or y.shape != expected:
                raise ValueError(f"Unexpected native image/mask size: {path.name}")
            allowed = {0, 20, 40} if domain == "cvd" else {1, 2, 3}
            if not set(np.unique(y)) <= allowed:
                raise ValueError(f"Unexpected mask encoding: {mask.name}")
            if domain == "cvd":
                y = y.astype(np.int64) // 20
            else:
                # Released source implementation crops; no resampling.
                x, y = x[:96, :96], y[:96, :96].astype(np.int64) - 1
            self.rows.append((path.stem, torch.from_numpy(x.transpose(2, 0, 1).copy()),
                              torch.from_numpy(y.copy())))
            self.records.append({"name": path.stem, "image": path.relative_to(root).as_posix(),
                                 "mask": mask.relative_to(root).as_posix(),
                                 "image_sha256": sha256(path), "mask_sha256": sha256(mask)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, x, y = self.rows[index]
        x = x.float() / 255
        if self.training:
            x, y = augment(x, y)
        return name, x, y
