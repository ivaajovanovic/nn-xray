import argparse
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


def save_feature_grid(feat: torch.Tensor, out_path: Path, max_channels: int = 16):
    feat = feat[0]  # (C, H, W)
    C, H, W = feat.shape
    n = min(C, max_channels)

    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        fm = feat[i].detach().cpu().numpy()

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"[ERROR] Image not found: {img_path}")

    out_dir = Path(args.out_dir)
    features_dir = out_dir / "feature_maps"
    debug_dir = out_dir / "debug"
    features_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, preprocess = load_model(device)

    activations = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def make_hook(layer_name):
                def hook_fn(_m, _inp, out):
                    activations[layer_name] = out
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"Registered hooks on {len(hooks)} Conv2d layers")

    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(x)

    saved = 0
    for layer_name in sorted(activations.keys()):
        feat = activations[layer_name]
        out_file = features_dir / f"{saved:02d}_{layer_name.replace('.', '_')}.png"
        save_feature_grid(feat, out_file, max_channels=16)
        saved += 1

    for h in hooks:
        h.remove()

    summary_path = debug_dir / "feature_maps_ok.txt"
    lines = [f"Total conv layers captured: {len(activations)}", "Layers:"]
    for k in sorted(activations.keys()):
        lines.append(f"- {k}: {tuple(activations[k].shape)}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {saved} feature-map grids to {features_dir}")
    print(f"Saved: {summary_path}")
    print("=== FEATURE MAPS DONE ===")


if __name__ == "__main__":
    main()
