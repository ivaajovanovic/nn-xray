import argparse
from pathlib import Path
import csv

import torch
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


def main():
    print("=== ACTIVATION STATS CHECK ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"[ERROR] Image not found: {img_path}")

    out_dir = Path(args.out_dir)
    stats_dir = out_dir / "activation_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, preprocess = load_model(device)

    # 1) hook: skupimo izlaze konv slojeva
    activations = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def make_hook(layer_name):
                def hook_fn(_module, _inp, out):
                    activations[layer_name] = out.detach()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name)))

    # 2) forward
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(x)

    # 3) statistike
    rows = []
    eps = 1e-6

    for layer_name in sorted(activations.keys()):
        feat = activations[layer_name]  # (1,C,H,W)
        _, C, H, W = feat.shape

        a = feat.abs()
        mean_abs = float(a.mean().cpu())
        max_abs = float(a.max().cpu())
        sparsity = float((a < eps).float().mean().cpu())

        rows.append({
            "layer": layer_name,
            "C": C, "H": H, "W": W,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "sparsity": sparsity
        })

    # 4) snimi CSV
    csv_path = stats_dir / "activation_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 5) graf
    x_idx = list(range(len(rows)))
    mean_abs_vals = [r["mean_abs"] for r in rows]
    sparsity_vals = [r["sparsity"] for r in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(x_idx, mean_abs_vals, marker="o", label="mean(abs)")
    plt.plot(x_idx, sparsity_vals, marker="o", label="sparsity (|a|<1e-6)")
    plt.xticks(x_idx, [str(i) for i in x_idx], rotation=0)
    plt.xlabel("Conv layer index (sorted by name)")
    plt.ylabel("Value")
    plt.title("Activation stats across conv layers")
    plt.legend()
    plt.tight_layout()

    plot_path = stats_dir / "activation_stats.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    # cleanup
    for h in hooks:
        h.remove()

    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")
    print("=== ACTIVATION STATS DONE ===")


if __name__ == "__main__":
    main()
