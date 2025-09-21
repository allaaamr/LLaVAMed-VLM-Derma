# src/models/baseline.py
"""
WHY: Swap CNN/ViT by name without changing training code.
WHAT: build_timm_model(name, num_classes)
"""
import timm, torch.nn as nn

def build_timm_model(name: str, num_classes: int) -> nn.Module:
    return timm.create_model(name, pretrained=True, num_classes=num_classes)
