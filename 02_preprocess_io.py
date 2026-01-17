import os
from pathlib import Path

import torch
from torchvision import models
from PIL import Image


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=== PREPROCESS IO CHECK ===")
    os.makedirs("outputs/debug", exist_ok=True)
    os.makedirs("outputs/input", exist_ok=True)

    # 1) Putanja do slike (stavi input.jpg u folder projekta)
    img_path = Path("input.jpg")
    if not img_path.exists():
        print("[ERROR] Ne postoji input.jpg u folderu projekta.")
        print("✅ Rešenje: stavi neku sliku u isti folder i nazovi je input.jpg")
        return

    # 2) Load image
    img = Image.open(img_path).convert("RGB")
    img.save("outputs/input/original.png")
    print("Saved: outputs/input/original.png")

    # 3) Preprocess (koristimo transform iz ResNet18 weights)
    weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    x = preprocess(img)              # Tensor (3, 224, 224) normalizovan
    x_batched = x.unsqueeze(0)       # (1, 3, 224, 224)

    # 4) Snimi "preprocessed preview" (bez normalize da se vidi slika)
    #    weights.transforms() uključuje normalize, pa za preview uradimo poseban resize/crop bez normalize.
    preview_tf = models.ResNet18_Weights.DEFAULT.transforms(antialias=True)
    # Preview tf je isti kao preprocess, ali i dalje uključuje normalize.
    # Zato pravimo ručno preview bez normalize:
    from torchvision import transforms
    preview_only = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    preview_img = preview_only(img)
    preview_img.save("outputs/input/preprocessed_224.png")
    print("Saved: outputs/input/preprocessed_224.png")

    # 5) Debug info o tensoru
    debug_path = Path("outputs/debug/preprocess_ok.txt")
    debug_path.write_text(
        f"PREPROCESS OK\n"
        f"Image: {img_path}\n"
        f"Tensor shape: {tuple(x.shape)}\n"
        f"Batched shape: {tuple(x_batched.shape)}\n"
        f"Tensor dtype: {x.dtype}\n"
        f"Tensor min: {float(x.min())}\n"
        f"Tensor max: {float(x.max())}\n",
        encoding="utf-8"
    )
    print(f"Saved: {debug_path}")
    print("=== PREPROCESS DONE ✅ ===")


if __name__ == "__main__":
    main()
