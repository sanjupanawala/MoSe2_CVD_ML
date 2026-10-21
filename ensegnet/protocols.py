"""Training settings used by the CVD and separate exfoliated-data experiments."""
from copy import deepcopy

PROTOCOLS = {
    "cvd": {
        "protocol": "cvd", "domain": "cvd",
        "counts": {"train": 97, "val": 12, "test": 12},
        "seeds": [20261010, 20261011, 20261012], "epochs": 50,
        "batch_size": 4, "validation_batch_size": 2, "learning_rate": 0.0001,
        "amp": True, "augmentation": True,
        "loss": "cross entropy plus macro soft Dice, equal coefficients",
        "selection": "maximum pooled validation mIoU; exact ties use earliest epoch",
    },
    "exfoliated": {
        "protocol": "exfoliated", "domain": "exfoliated",
        "counts": {"train": 1024, "val": 256, "test": 320},
        "seeds": [20261010, 20261011, 20261012], "epochs": 50,
        "batch_size": 4, "validation_batch_size": 16, "learning_rate": 0.0001,
        "amp": True, "augmentation": True,
        "loss": "cross entropy plus macro soft Dice, equal coefficients",
        "selection": "maximum pooled validation mIoU; exact ties use earliest epoch",
    },
    "leger-source": {
        "protocol": "leger-source", "domain": "exfoliated",
        "counts": {"train": 1024, "val": 256, "test": 320},
        "seeds": [20261008], "epochs": 20,
        "batch_size": 4, "validation_batch_size": 4, "learning_rate": 0.0001,
        "amp": False, "augmentation": False,
        "loss": "cross entropy; grayscale 0..255 as in released implementation",
        "selection": "maximum pooled validation mIoU; exact ties use earliest epoch",
    },
}


def get_protocol(name):
    return deepcopy(PROTOCOLS[name])
