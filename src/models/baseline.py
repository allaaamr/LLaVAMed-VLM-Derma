# src/models/baseline.py
"""
WHY: Swap CNN/ViT by name without changing training code.
WHAT: build_timm_model(name, num_classes)
"""
import timm, torch.nn as nn

def build_timm_model(name: str, num_classes: int, drop_rate: float = 0.2) -> nn.Module:
    return timm.create_model(name, pretrained=True, num_classes=num_classes, drop_rate=drop_rate)
