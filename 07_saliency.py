import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    labels = weights.meta.get("categories", None)
    return model, preprocess, labels


def main():
    print("=== SALIENCY CHECK ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"[ERROR] Image not found: {img_path}")

    out_dir = Path(args.out_dir)
    saliency_dir = out_dir / "saliency"
    saliency_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, preprocess, labels = load_model(device)

    # image + input tensor
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # forward
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    top_idx = int(torch.argmax(probs).item())
    top_prob = float(probs[top_idx].item())
    top_name = labels[top_idx] if labels else f"class_{top_idx}"
    print(f"Top-1: {top_name} (id={top_idx}) prob={top_prob:.4f}")

    # backward: grad w.r.t. input
    model.zero_grad(set_to_none=True)
    score = logits[0, top_idx]
    score.backward()

    # saliency: |grad| aggregated over channels
    grad = x.grad.detach()[0]          # (3,224,224)
    sal = grad.abs().mean(dim=0)       # (224,224)

    # normalize 0..1
    sal = sal - sal.min()
    if sal.max() > 0:
        sal = sal / sal.max()
    sal_np = sal.cpu().numpy()

    # create 224 preview image
    preview_only = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    img_224 = preview_only(img)
    img_224_np = np.array(img_224).astype(np.float32) / 255.0

    # save saliency map
    map_path = saliency_dir / "saliency_map.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(sal_np, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(map_path, dpi=160)
    plt.close()

    # overlay
    overlay_path = saliency_dir / "saliency_overlay.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(img_224_np)
    plt.imshow(sal_np, cmap="jet", alpha=0.45)
    plt.title(f"{top_name} ({top_prob:.3f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=160)
    plt.close()

    print(f"Saved: {map_path}")
    print(f"Saved: {overlay_path}")
    print("=== SALIENCY DONE ===")


if __name__ == "__main__":
    main()
