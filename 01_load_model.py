import os
from pathlib import Path

import torch
from torchvision import models


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs("outputs/debug", exist_ok=True)

    device = get_device()
    print("=== LOAD MODEL CHECK ===")
    print(f"Device: {device}")

    # 1) Load pretrained ResNet18 (ImageNet)
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device)
    model.eval()

    # 2) Dummy forward (provera da sve radi)
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)

    # 3) Check output shape
    print(f"Output shape: {tuple(out.shape)} (expected: (1, 1000))")

    # 4) Save a tiny debug file
    info_path = Path("outputs/debug/model_ok.txt")
    info_path.write_text(
        f"MODEL OK\n"
        f"Model: resnet18 (pretrained ImageNet)\n"
        f"Device: {device}\n"
        f"Output shape: {tuple(out.shape)}\n",
        encoding="utf-8"
    )

    print(f"Saved: {info_path}")
    print("=== MODEL LOAD DONE ✅ ===")


if __name__ == "__main__":
    main()
