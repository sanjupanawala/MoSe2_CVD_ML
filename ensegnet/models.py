"""Final ensemble members and the two locally implemented comparators.

Module names deliberately match the archived checkpoints. No training weights
are downloaded when constructing a model for inference.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

MEMBERS = ("segformer_b1", "student_control", "scse")
LABELS = ("Substrate", "Monolayer", "Multilayer")


class UNet(nn.Module):
    def __init__(self, kind, pretrained=False, source_gray=False):
        super().__init__()
        self.kind = kind
        self.source_gray = source_gray
        encoder = {"student_control": "resnet18", "scse": "resnet50", "leger": "resnet34"}[kind]
        self.net = smp.Unet(
            encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
            in_channels=1 if kind == "leger" else 3, classes=3, activation=None,
            decoder_attention_type="scse" if kind == "scse" else None,
        )

    def forward(self, x):
        if self.kind == "leger":
            # Source reproduction used grayscale on the JPEG 0..255 scale;
            # the CVD retraining used grayscale after RGB / 255.
            if self.source_gray:
                rgb = (x.double() * 255).round()
                x = (rgb[:, :1] * .2989 + rgb[:, 1:2] * .5870 + rgb[:, 2:3] * .1140).float()
            else:
                x = x[:, :1] * .2989 + x[:, 1:2] * .5870 + x[:, 2:3] * .1140
        return {"logits": self.net(x)}


class Segformer(nn.Module):
    def __init__(self, pretrained=False, asset="nvidia/mit-b1"):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel
        # MiT-B1, including the 256-channel decoder used in the saved experiment.
        config = SegformerConfig(
            hidden_sizes=[64, 128, 320, 512], depths=[2, 2, 2, 2],
            decoder_hidden_size=256, num_labels=3, semantic_loss_ignore_index=255,
        )
        self.net = SegformerForSemanticSegmentation(config)
        if pretrained:
            encoder = SegformerModel.from_pretrained(str(asset))
            self.net.segformer.load_state_dict(encoder.state_dict(), strict=True)

    def forward(self, x):
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        logits = self.net(pixel_values=(x - mean) / std).logits
        return {"logits": F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)}


def _convs(channels):
    layers = []
    for a, b in zip(channels[:-1], channels[1:]):
        layers.extend([nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)])
    return nn.Sequential(*layers)


class Han(nn.Module):
    """VGG16/index-unpool reconstruction, not the authors' final checkpoint."""
    def __init__(self, pretrained=False, asset=None):
        super().__init__()
        self.encoder = nn.ModuleList([_convs(c) for c in [
            [3, 64, 64], [64, 128, 128], [128, 256, 256, 256],
            [256, 512, 512, 512], [512, 512, 512, 512],
        ]])
        self.decoder = nn.ModuleList([_convs(c) for c in [
            [512, 512, 512, 512], [512, 512, 512, 256],
            [256, 256, 256, 128], [128, 128, 64], [64, 64, 64],
        ]])
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.classifier = nn.Conv2d(64, 3, 1)
        if pretrained:
            state = (torch.load(asset, map_location="cpu", weights_only=True) if asset else
                     torch.hub.load_state_dict_from_url(
                         "https://download.pytorch.org/models/vgg16-397923af.pth", check_hash=True))
            indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
            convs = [m for m in self.encoder.modules() if isinstance(m, nn.Conv2d)]
            with torch.no_grad():
                for module, index in zip(convs, indices):
                    module.weight.copy_(state[f"features.{index}.weight"])
                    module.bias.copy_(state[f"features.{index}.bias"])

    def forward(self, x):
        indices, sizes = [], []
        for block in self.encoder:
            x = block(x)
            sizes.append(x.size())
            x, index = self.pool(x)
            indices.append(index)
        for block, index, size in zip(self.decoder, reversed(indices), reversed(sizes)):
            x = block(self.unpool(x, index, output_size=size))
        return {"logits": self.classifier(x)}


def make_model(kind, pretrained=False, asset=None, source_gray=False):
    if kind == "segformer_b1":
        return Segformer(pretrained, asset or "nvidia/mit-b1")
    if kind == "han":
        return Han(pretrained, asset)
    return UNet(kind, pretrained, source_gray)


def load_model(path, kind, device="cpu", source_gray=False):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = payload.get("state", payload)
    model = make_model(kind, source_gray=source_gray)
    # The source Leger run saved smp.Unet directly; CVD runs saved a wrapper.
    if kind == "leger" and not any(k.startswith("net.") for k in state):
        state = {"net." + k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
