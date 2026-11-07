"""Fine-grained classifier: timm backbone + 196-way class head + auxiliary make head."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms as T

from src import config


class CarClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int = config.NUM_CLASSES, num_makes: int = 49,
                 pretrained: bool = True, drop_rate: float = 0.2, drop_path_rate: float = 0.1):
        super().__init__()
        import timm

        kwargs = {"drop_path_rate": drop_path_rate} if not backbone.startswith("resnet") else {}
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kwargs)
        dim = self.backbone.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.class_head = nn.Linear(dim, num_classes)
        self.make_head = nn.Linear(dim, num_makes) if num_makes > 0 else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # pooled embedding, (B, dim)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        emb = self.forward_features(x)
        h = self.dropout(emb)
        logits = self.class_head(h)
        if not return_all:
            return logits
        make_logits = self.make_head(h) if self.make_head is not None else None
        return logits, make_logits, emb


def build_transforms(img_size: int, train: bool):
    norm = T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    if not train:
        return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.ToTensor(),
        norm,
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def save_checkpoint(path: Path, model: CarClassifier, args: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": args, "metrics": metrics}, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[CarClassifier, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = CarClassifier(a["backbone"], num_makes=a["num_makes"] if a["make_weight"] > 0 else 0,
                          pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
