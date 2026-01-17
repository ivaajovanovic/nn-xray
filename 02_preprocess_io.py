import argparse
from pathlib import Path

from PIL import Image
from torchvision import models, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out_dir", required=True, help="Per-image output directory")
    args = parser.parse_args()

    print("=== PREPROCESS IO CHECK ===")

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"[ERROR] Image not found: {img_path}")

    out_dir = Path(args.out_dir)
    input_dir = out_dir / "input"
    debug_dir = out_dir / "debug"
    input_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load image
    img = Image.open(img_path).convert("RGB")

    # Save original copy (png)
    original_path = input_dir / "original.png"
    img.save(original_path)
    print(f"Saved: {original_path}")

    # 2) Preprocess (ResNet18 default transforms)
    weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    x = preprocess(img)        # Tensor (3, 224, 224) normalized
    x_batched = x.unsqueeze(0) # (1, 3, 224, 224)

    # 3) Preprocessed preview (without normalize) for UI
    preview_only = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    preview_img = preview_only(img)

    preprocessed_path = input_dir / "preprocessed_224.png"
    preview_img.save(preprocessed_path)
    print(f"Saved: {preprocessed_path}")

    # 4) Debug info
    debug_path = debug_dir / "preprocess_ok.txt"
    debug_path.write_text(
        f"PREPROCESS OK\n"
        f"Image: {img_path}\n"
        f"Tensor shape: {tuple(x.shape)}\n"
        f"Batched shape: {tuple(x_batched.shape)}\n"
        f"Tensor dtype: {x.dtype}\n"
        f"Tensor min: {float(x.min())}\n"
        f"Tensor max: {float(x.max())}\n",
        encoding="utf-8",
    )
    print(f"Saved: {debug_path}")
    print("=== PREPROCESS DONE ===")


if __name__ == "__main__":
    main()
