import os
from pathlib import Path
import json
import csv

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt


# ----------------- basic utils -----------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    labels = weights.meta.get("categories", None)
    return model, preprocess, labels


def to_uint8(img01):
    return (np.clip(img01, 0, 1) * 255).astype(np.uint8)


def unnormalize_to_img224(x_norm_chw):
    # x_norm_chw: (3,224,224) normalized
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = x_norm_chw * std + mean
    x = np.clip(x, 0, 1)
    return np.transpose(x, (1, 2, 0))  # (224,224,3)


@torch.no_grad()
def predict_probs(model, x):
    logits = model(x)
    return F.softmax(logits, dim=1)[0]


# ----------------- Grad-CAM -----------------
def gradcam(model, x, target_layer, class_idx):
    activ = None
    grad = None

    def fwd_hook(_m, _inp, out):
        nonlocal activ
        activ = out

    def bwd_hook(_m, gin, gout):
        nonlocal grad
        grad = gout[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    A = activ.detach()  # (1,C,H,W)
    G = grad.detach()   # (1,C,H,W)

    w = G.mean(dim=(2, 3), keepdim=True)           # (1,C,1,1)
    cam = (w * A).sum(dim=1)                       # (1,H,W)
    cam = F.relu(cam)[0]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_up = F.interpolate(cam[None, None, :, :], size=(224, 224),
                           mode="bilinear", align_corners=False)[0, 0]

    h1.remove()
    h2.remove()

    return cam_up.detach().cpu().numpy()  # (224,224) in [0,1]


# ----------------- Saliency (vanilla gradient) -----------------
def saliency_map(model, x, class_idx):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    g = x.grad.detach()[0]                    # (3,224,224)
    g = g.abs().max(dim=0).values             # (224,224)
    g = g - g.min()
    if g.max() > 0:
        g = g / g.max()
    return g.cpu().numpy()  # (224,224) in [0,1]


# ----------------- Feature maps (selected checkpoints) -----------------
@torch.no_grad()
def forward_selected_featuremaps(model, x):
    # return dict name -> tensor (1,C,H,W)
    feats = {}

    z = model.conv1(x)
    z = model.bn1(z)
    z = model.relu(z)
    feats["stem_conv1"] = z
    z = model.maxpool(z)

    z = model.layer1(z)
    feats["layer1_out"] = z
    z = model.layer2(z)
    feats["layer2_out"] = z
    z = model.layer3(z)
    feats["layer3_out"] = z
    z = model.layer4(z)
    feats["layer4_out"] = z

    return feats


def save_featuremap_grid(z, out_path, max_channels=16):
    """
    z: (1,C,H,W)
    saves a grid of first N channels (normalized per-channel)
    """
    z = z.detach().cpu()[0]   # (C,H,W)
    C = z.shape[0]
    n = min(C, max_channels)

    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 2.0, rows * 2.0))
    for i in range(n):
        fm = z[i].numpy()
        fm = fm - fm.min()
        if fm.max() > 0:
            fm = fm / fm.max()

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(fm, cmap="gray")
        ax.axis("off")
        ax.set_title(f"ch{i}", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ----------------- Activation stats across ALL conv layers -----------------
@torch.no_grad()
def activation_stats_conv_layers(model, x):
    stats = []  # list of dicts: name, mean_abs, sparsity
    hooks = []

    def make_hook(name):
        def hook(_m, _inp, out):
            # out: (1,C,H,W)
            a = out.detach()
            mean_abs = float(a.abs().mean().cpu())
            sparsity = float((a.abs() < 1e-6).float().mean().cpu())
            stats.append({"layer": name, "mean_abs": mean_abs, "sparsity": sparsity})
        return hook

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(name)))

    _ = model(x)

    for h in hooks:
        h.remove()

    # stable order
    stats.sort(key=lambda d: d["layer"])
    return stats


def save_activation_stats(stats, csv_path, png_path):
    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "mean_abs", "sparsity"])
        w.writeheader()
        w.writerows(stats)

    # Plot
    xs = list(range(len(stats)))
    mean_abs = [s["mean_abs"] for s in stats]
    sparsity = [s["sparsity"] for s in stats]

    plt.figure(figsize=(10, 3.8))
    plt.plot(xs, mean_abs, marker="o", label="mean(|act|)")
    plt.plot(xs, sparsity, marker="o", label="sparsity(|a|<1e-6)")
    plt.xticks(xs, [s["layer"] for s in stats], rotation=90, fontsize=6)
    plt.ylabel("value")
    plt.title("Activation stats across conv layers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


# ----------------- Confidence trace (non-flat, NO training) -----------------
def gate_topk_channels(z, keep_ratio=0.25):
    # z: (1,C,H,W)
    C = z.shape[1]
    k = max(1, int(C * keep_ratio))
    score = z.abs().mean(dim=(0, 2, 3))          # (C,)
    topk = torch.topk(score, k).indices
    mask = torch.zeros(C, device=z.device)
    mask[topk] = 1.0
    return z * mask.view(1, C, 1, 1)


@torch.no_grad()
def forward_checkpoints_resnet18(model, x):
    acts = {}

    z = model.conv1(x)
    z = model.bn1(z)
    z = model.relu(z)
    z = model.maxpool(z)
    acts["stem"] = z

    z = model.layer1(z); acts["layer1"] = z
    z = model.layer2(z); acts["layer2"] = z
    z = model.layer3(z); acts["layer3"] = z
    z = model.layer4(z); acts["layer4"] = z

    return acts


@torch.no_grad()
def forward_from_checkpoint(model, z, start_cp):
    if start_cp == "stem":
        z = model.layer1(z); z = model.layer2(z); z = model.layer3(z); z = model.layer4(z)
    elif start_cp == "layer1":
        z = model.layer2(z); z = model.layer3(z); z = model.layer4(z)
    elif start_cp == "layer2":
        z = model.layer3(z); z = model.layer4(z)
    elif start_cp == "layer3":
        z = model.layer4(z)
    elif start_cp == "layer4":
        pass
    else:
        raise ValueError("start_cp must be: stem/layer1/layer2/layer3/layer4")

    pooled = model.avgpool(z)
    pooled = torch.flatten(pooled, 1)
    logits = model.fc(pooled)
    probs = F.softmax(logits, dim=1)[0]
    return probs


def save_confidence_trace(model, x, labels, out_dir: Path, keep_ratio=0.25):
    probs_final = predict_probs(model, x)
    target_idx = int(torch.argmax(probs_final).item())
    target_prob = float(probs_final[target_idx].item())
    target_name = labels[target_idx] if labels else f"class_{target_idx}"

    acts = forward_checkpoints_resnet18(model, x)
    checkpoints = ["stem", "layer1", "layer2", "layer3", "layer4", "final"]

    rows = []
    for cp in checkpoints:
        if cp == "final":
            p = float(probs_final[target_idx].item())
        else:
            z = acts[cp]
            z_g = gate_topk_channels(z, keep_ratio=keep_ratio)
            probs = forward_from_checkpoint(model, z_g, cp)
            p = float(probs[target_idx].item())

        rows.append({"checkpoint": cp, "p_target": p})

    # CSV/TXT
    (out_dir / "confidence_trace.csv").write_text(
        "checkpoint,p_target\n" + "\n".join([f"{r['checkpoint']},{r['p_target']}" for r in rows]),
        encoding="utf-8"
    )
    (out_dir / "confidence_trace.txt").write_text(
        f"Target: {target_name} (id={target_idx}) final_prob={target_prob:.4f}\n"
        f"gate_topk_channels keep_ratio={keep_ratio}\n\n" +
        "\n".join([f"{r['checkpoint']}: {r['p_target']:.6f}" for r in rows]),
        encoding="utf-8"
    )

    # Plot
    xs = list(range(len(rows)))
    plt.figure(figsize=(9, 3.8))
    plt.plot(xs, [r["p_target"] for r in rows], marker="o")
    plt.xticks(xs, [r["checkpoint"] for r in rows])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Checkpoint (after block)")
    plt.ylabel("P(target class) with channel-gating")
    plt.title(f"Confidence trace for target: {target_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_trace.png", dpi=160)
    plt.close()

    return {"id": target_idx, "label": target_name, "prob": target_prob}


# ----------------- Two-image comparison (A vs B) -----------------
def save_two_image_comparison(model, preprocess, labels, imgA_path, imgB_path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    imgA = Image.open(imgA_path).convert("RGB")
    imgB = Image.open(imgB_path).convert("RGB")

    xA = preprocess(imgA).unsqueeze(0).to(next(model.parameters()).device)
    xB = preprocess(imgB).unsqueeze(0).to(next(model.parameters()).device)

    probsA = predict_probs(model, xA)
    probsB = predict_probs(model, xB)

    idA = int(torch.argmax(probsA).item())
    idB = int(torch.argmax(probsB).item())

    lblA = labels[idA] if labels else f"class_{idA}"
    lblB = labels[idB] if labels else f"class_{idB}"

    pA = float(probsA[idA].item())
    pB = float(probsB[idB].item())

    # Grad-CAM overlays
    target_layer = model.layer4[1].conv2
    xA_req = xA.clone().detach().requires_grad_(True)
    xB_req = xB.clone().detach().requires_grad_(True)

    camA = gradcam(model, xA_req, target_layer, idA)
    camB = gradcam(model, xB_req, target_layer, idB)

    imgA224 = unnormalize_to_img224(xA.detach().cpu()[0].numpy())
    imgB224 = unnormalize_to_img224(xB.detach().cpu()[0].numpy())

    heatA = np.stack([camA, camA**2, 1 - camA], axis=-1)
    heatB = np.stack([camB, camB**2, 1 - camB], axis=-1)

    heatA = (heatA - heatA.min()) / (heatA.max() - heatA.min() + 1e-8)
    heatB = (heatB - heatB.min()) / (heatB.max() - heatB.min() + 1e-8)

    overA = np.clip(0.55 * imgA224 + 0.45 * heatA, 0, 1)
    overB = np.clip(0.55 * imgB224 + 0.45 * heatB, 0, 1)

    # One figure side-by-side
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(overA)
    plt.axis("off")
    plt.title(f"A: {imgA_path.name}\nTop-1: {lblA} ({pA:.3f})")

    plt.subplot(1, 2, 2)
    plt.imshow(overB)
    plt.axis("off")
    plt.title(f"B: {imgB_path.name}\nTop-1: {lblB} ({pB:.3f})")

    plt.tight_layout()
    plt.savefig(out_dir / "compare_gradcam.png", dpi=160)
    plt.close()

    # Confidence traces (channel-gated)
    traceA_dir = out_dir / "_traceA"
    traceB_dir = out_dir / "_traceB"
    traceA_dir.mkdir(exist_ok=True)
    traceB_dir.mkdir(exist_ok=True)

    targetA = save_confidence_trace(model, xA, labels, traceA_dir, keep_ratio=0.25)
    targetB = save_confidence_trace(model, xB, labels, traceB_dir, keep_ratio=0.25)

    # Load CSVs and plot together
    def read_trace_csv(p):
        lines = (p / "confidence_trace.csv").read_text(encoding="utf-8").strip().splitlines()[1:]
        cp = []
        vals = []
        for ln in lines:
            a, b = ln.split(",")
            cp.append(a)
            vals.append(float(b))
        return cp, vals

    cpA, vA = read_trace_csv(traceA_dir)
    cpB, vB = read_trace_csv(traceB_dir)

    xs = list(range(len(cpA)))
    plt.figure(figsize=(10, 4))
    plt.plot(xs, vA, marker="o", label=f"A: {imgA_path.name}")
    plt.plot(xs, vB, marker="o", label=f"B: {imgB_path.name}")
    plt.xticks(xs, cpA)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Checkpoint (after block)")
    plt.ylabel("P(target class) with channel-gating")
    plt.title("Confidence trace (keep_ratio=0.25)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_confidence_trace.png", dpi=160)
    plt.close()

    # Summary
    (out_dir / "compare_summary.txt").write_text(
        f"A: {imgA_path.name} -> {lblA} (p={pA:.4f})\n"
        f"B: {imgB_path.name} -> {lblB} (p={pB:.4f})\n\n"
        f"Interpretation hint:\n"
        f"- A is 'easier' if its confidence rises earlier + ends high/stable.\n"
        f"- B is 'harder' if confidence stays low longer / fluctuates / ends lower.\n",
        encoding="utf-8"
    )

    # cleanup temp subdirs (optional)
    # leave them; can be useful for debugging


# ----------------- main batch -----------------
def main():
    print("=== BATCH REPORT (FULL) CHECK ===")

    in_dir = Path("inputs")
    if not in_dir.exists():
        in_dir.mkdir(parents=True, exist_ok=True)
        print("[INFO] Kreiran folder inputs/. Ubaci slike (jpg/png) pa pokreni opet.")
        return

    out_root = Path("outputs/report")
    out_root.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, preprocess, labels = load_model(device)

    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        img_files += list(in_dir.glob(ext))
    img_files = sorted(img_files)

    if not img_files:
        print("[INFO] Nema slika u inputs/. Ubaci par slika pa pokreni opet.")
        return

    # per-image reports
    for img_path in img_files:
        name = img_path.stem
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)

        # ----- prediction
        probs = predict_probs(model, x).detach().cpu()
        top5 = torch.topk(probs, 5)
        top5_ids = top5.indices.tolist()
        top5_probs = top5.values.tolist()

        top1_id = top5_ids[0]
        top1_prob = top5_probs[0]
        top1_name = labels[top1_id] if labels else f"class_{top1_id}"

        pred_json = {
            "file": img_path.name,
            "top1": {"id": top1_id, "label": top1_name, "prob": float(top1_prob)},
            "top5": [
                {
                    "id": int(i),
                    "label": labels[int(i)] if labels else f"class_{int(i)}",
                    "prob": float(p),
                }
                for i, p in zip(top5_ids, top5_probs)
            ],
        }
        (out_dir / "prediction.json").write_text(json.dumps(pred_json, indent=2), encoding="utf-8")

        lines = [f"Top-1: {top1_name} (id={top1_id}) prob={top1_prob:.4f}", "", "Top-5:"]
        for i, p in zip(top5_ids, top5_probs):
            lbl = labels[int(i)] if labels else f"class_{int(i)}"
            lines.append(f"- {lbl} (id={int(i)}): {p:.4f}")
        (out_dir / "prediction.txt").write_text("\n".join(lines), encoding="utf-8")

        # ----- model view 224 (for overlays)
        x0 = x.detach().cpu()[0].numpy()  # (3,224,224) normalized
        img224 = unnormalize_to_img224(x0)
        Image.fromarray(to_uint8(img224)).save(out_dir / "model_view_224.png")

        # ----- Grad-CAM
        target_layer = model.layer4[1].conv2
        x_req = x.clone().detach().requires_grad_(True)
        cam = gradcam(model, x_req, target_layer, top1_id)

        heat = np.stack([cam, cam**2, 1 - cam], axis=-1)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        overlay = np.clip(0.55 * img224 + 0.45 * heat, 0, 1)

        Image.fromarray(to_uint8(heat)).save(out_dir / "gradcam_heatmap.png")
        Image.fromarray(to_uint8(overlay)).save(out_dir / "gradcam_overlay.png")

        # ----- Saliency
        sal = saliency_map(model, x, top1_id)
        sal_rgb = np.stack([sal, sal**2, 1 - sal], axis=-1)
        sal_rgb = (sal_rgb - sal_rgb.min()) / (sal_rgb.max() - sal_rgb.min() + 1e-8)
        sal_overlay = np.clip(0.55 * img224 + 0.45 * sal_rgb, 0, 1)

        Image.fromarray(to_uint8(sal)).save(out_dir / "saliency_map.png")
        Image.fromarray(to_uint8(sal_overlay)).save(out_dir / "saliency_overlay.png")

        # ----- Feature maps (selected)
        fm_dir = out_dir / "feature_maps"
        fm_dir.mkdir(exist_ok=True)
        feats = forward_selected_featuremaps(model, x)
        for k, z in feats.items():
            save_featuremap_grid(z, fm_dir / f"featuremaps_{k}.png", max_channels=16)

        # ----- Activation stats (all conv layers)
        stats = activation_stats_conv_layers(model, x)
        save_activation_stats(stats, out_dir / "activation_stats.csv", out_dir / "activation_stats.png")

        # ----- Confidence trace
        trace_dir = out_dir / "trace"
        trace_dir.mkdir(exist_ok=True)
        _target = save_confidence_trace(model, x, labels, trace_dir, keep_ratio=0.25)

        print(f"[OK] {img_path.name} -> {top1_name} ({top1_prob:.3f}) | saved to {out_dir}")

    # two-image comparison: first two images (or choose your own by renaming)
    if len(img_files) >= 2:
        imgA = img_files[0]
        imgB = img_files[1]
        comp_dir = out_root / "_compare" / f"{imgA.stem}__vs__{imgB.stem}"
        save_two_image_comparison(model, preprocess, labels, imgA, imgB, comp_dir)
        print(f"[OK] Two-image comparison saved to: {comp_dir}")

    print("=== BATCH REPORT DONE ✅ ===")
    print("Ubaci slike u inputs/ i pogledaj outputs/report/<ime_slike>/ ...")


if __name__ == "__main__":
    main()
