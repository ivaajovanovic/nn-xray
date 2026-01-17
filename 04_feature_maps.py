import os
from pathlib import Path

import torch
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


def ensure_dirs():
    os.makedirs("outputs/features", exist_ok=True)
    os.makedirs("outputs/debug", exist_ok=True)


def save_feature_grid(feat: torch.Tensor, out_path: Path, max_channels: int = 16):
    """
    feat: (1, C, H, W)
    snima grid prvih max_channels kanala kao jednu sliku.
    """
    feat = feat[0]  # (C, H, W)
    C, H, W = feat.shape
    n = min(C, max_channels)

    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        fm = feat[i].detach().cpu().numpy()

        # normalizacija da izgleda lepo (0..1)
        fm = fm - fm.min()
        if fm.max() > 0:
            fm = fm / fm.max()

        ax.imshow(fm, cmap="gray")
        ax.set_title(f"ch {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    print("=== FEATURE MAPS CHECK ===")
    ensure_dirs()

    img_path = Path("input.jpg")
    if not img_path.exists():
        print("[ERROR] Ne postoji input.jpg")
        return

    device = get_device()
    model, preprocess = load_model(device)

    # 1) Hook storage
    activations = {}  # name -> tensor

    # 2) Register hooks for Conv2d layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def make_hook(layer_name):
                def hook_fn(_module, _inp, out):
                    activations[layer_name] = out
                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"Registered hooks on {len(hooks)} Conv2d layers")

    # 3) Forward pass
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(x)

    # 4) Save feature grids
    # sortiramo po imenu da redosled bude stabilan
    saved = 0
    for layer_name in sorted(activations.keys()):
        feat = activations[layer_name]  # (1,C,H,W)
        out_file = Path("outputs/features") / f"{saved:02d}_{layer_name.replace('.', '_')}.png"
        save_feature_grid(feat, out_file, max_channels=16)
        saved += 1

    # 5) Cleanup hooks
    for h in hooks:
        h.remove()

    # 6) Debug summary
    summary_path = Path("outputs/debug/feature_maps_ok.txt")
    lines = [f"Total conv layers captured: {len(activations)}", "Layers:"]
    for k in sorted(activations.keys()):
        shape = tuple(activations[k].shape)
        lines.append(f"- {k}: {shape}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {saved} feature-map grids to outputs/features/")
    print(f"Saved: {summary_path}")
    print("=== FEATURE MAPS DONE ✅ ===")


if __name__ == "__main__":
    main()
