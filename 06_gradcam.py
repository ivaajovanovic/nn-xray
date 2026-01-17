import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models
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
    print("=== GRADCAM CHECK ===")
    os.makedirs("outputs/heatmaps", exist_ok=True)
    os.makedirs("outputs/debug", exist_ok=True)

    img_path = Path("input.jpg")
    if not img_path.exists():
        print("[ERROR] Ne postoji input.jpg")
        return

    device = get_device()
    model, preprocess, labels = load_model(device)

    # Target layer: poslednji conv u ResNet18
    target_layer = model.layer4[1].conv2

    # Hook storage
    activ = None
    grad = None

    def forward_hook(_m, _inp, out):
        nonlocal activ
        activ = out

    def backward_hook(_m, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Load + preprocess
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # Forward
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]

    top_idx = int(torch.argmax(probs).item())
    top_prob = float(probs[top_idx].item())
    top_name = labels[top_idx] if labels else f"class_{top_idx}"
    print(f"Top-1: {top_name} (id={top_idx}) prob={top_prob:.4f}")

    # Backward for that class
    model.zero_grad(set_to_none=True)
    score = logits[0, top_idx]
    score.backward()

    # Now we have activ: (1,C,H,W) and grad: (1,C,H,W)
    A = activ.detach()
    G = grad.detach()

    # Grad-CAM weights: global-average-pool gradients over spatial dims
    weights = G.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)

    cam = (weights * A).sum(dim=1, keepdim=False)  # (1,H,W)
    cam = F.relu(cam)
    cam = cam[0]  # (H,W)

    # Normalize to 0..1
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_np = cam.cpu().numpy()

    # We need an image to overlay: use center-crop 224 preview
    # (da se poklapa sa model inputom)
    from torchvision import transforms
    preview_only = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    img_224 = preview_only(img)
    img_224_np = np.array(img_224).astype(np.float32) / 255.0  # (224,224,3)

    # Upsample cam to 224x224
    cam_up = torch.tensor(cam_np)[None, None, :, :]  # (1,1,H,W)
    cam_up = F.interpolate(cam_up, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()

    # Save heatmap alone
    heatmap_path = Path("outputs/heatmaps/gradcam_heatmap.png")
    plt.figure(figsize=(4, 4))
    plt.imshow(cam_up, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=160)
    plt.close()

    # Overlay
    overlay_path = Path("outputs/heatmaps/gradcam_overlay.png")
    plt.figure(figsize=(4, 4))
    plt.imshow(img_224_np)
    plt.imshow(cam_up, cmap="jet", alpha=0.45)
    plt.title(f"{top_name} ({top_prob:.3f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=160)
    plt.close()

    # Cleanup hooks
    h1.remove()
    h2.remove()

    print(f"Saved: {heatmap_path}")
    print(f"Saved: {overlay_path}")
    print("=== GRADCAM DONE ✅ ===")


if __name__ == "__main__":
    main()
