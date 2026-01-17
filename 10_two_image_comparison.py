import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
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


@torch.no_grad()
def predict_probs(model, x):
    logits = model(x)
    return F.softmax(logits, dim=1)[0]


def unnormalize_to_img01(x_chw):
    # x_chw: (3,224,224) normalized
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x_chw.dtype, device=x_chw.device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=x_chw.dtype, device=x_chw.device)[:, None, None]
    img = (x_chw * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).detach().cpu().numpy()  # (224,224,3) in [0,1]


# ----------------- gradcam -----------------
def gradcam_map(model, x, target_layer, class_idx):
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


def cam_to_heat_rgb(cam):
    # Simple "jet-ish" heatmap without seaborn; output in [0,1]
    c = cam
    heat = np.stack([c, np.square(c), 1 - c], axis=-1)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat


# ----------------- confidence trace (gated channels) -----------------
@torch.no_grad()
def forward_checkpoints(model, x):
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


def gate_topk_channels(z, keep_ratio=0.25):
    C = z.shape[1]
    k = max(1, int(C * keep_ratio))
    score = z.abs().mean(dim=(0, 2, 3))
    topk = torch.topk(score, k).indices
    mask = torch.zeros(C, device=z.device)
    mask[topk] = 1.0
    return z * mask.view(1, C, 1, 1)


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
    return F.softmax(logits, dim=1)[0]


def confidence_trace_for_image(model, preprocess, img_pil, device, keep_ratio=0.25):
    x = preprocess(img_pil).unsqueeze(0).to(device)

    probs_final = predict_probs(model, x)
    target_idx = int(torch.argmax(probs_final).item())
    target_prob = float(probs_final[target_idx].item())

    acts = forward_checkpoints(model, x)
    cps = ["stem", "layer1", "layer2", "layer3", "layer4", "final"]
    trace = []

    for cp in cps:
        if cp == "final":
            p = float(probs_final[target_idx].item())
        else:
            z = acts[cp]
            z = gate_topk_channels(z, keep_ratio=keep_ratio)
            probs_cp = forward_from_checkpoint(model, z, cp)
            p = float(probs_cp[target_idx].item())
        trace.append(p)

    return {
        "x": x,
        "target_idx": target_idx,
        "target_prob": target_prob,
        "trace": trace,
        "checkpoints": cps
    }


# ----------------- main -----------------
def main():
    print("=== TWO IMAGE COMPARISON CHECK ===")
    os.makedirs("outputs/compare", exist_ok=True)

    in_dir = Path("inputs")
    if not in_dir.exists():
        in_dir.mkdir(parents=True, exist_ok=True)
        print("[INFO] Kreiran inputs/. Ubaci TACNO 2 slike i pokreni opet.")
        return

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        imgs += list(in_dir.glob(ext))
    imgs = sorted(imgs)

    if len(imgs) != 2:
        print(f"[ERROR] U inputs/ mora biti TACNO 2 slike. Trenutno: {len(imgs)}")
        for p in imgs:
            print(" -", p.name)
        return

    device = get_device()
    model, preprocess, labels = load_model(device)
    target_layer = model.layer4[1].conv2

    # Load images
    imgA = Image.open(imgs[0]).convert("RGB")
    imgB = Image.open(imgs[1]).convert("RGB")

    # Compute per-image: pred + trace
    keep_ratio = 0.25
    infoA = confidence_trace_for_image(model, preprocess, imgA, device, keep_ratio=keep_ratio)
    infoB = confidence_trace_for_image(model, preprocess, imgB, device, keep_ratio=keep_ratio)

    nameA = labels[infoA["target_idx"]] if labels else f"class_{infoA['target_idx']}"
    nameB = labels[infoB["target_idx"]] if labels else f"class_{infoB['target_idx']}"

    # Grad-CAM for each (for its own top-1)
    xA = infoA["x"].clone().detach().requires_grad_(True)
    camA = gradcam_map(model, xA, target_layer, infoA["target_idx"])
    imgA_224 = unnormalize_to_img01(xA[0])

    xB = infoB["x"].clone().detach().requires_grad_(True)
    camB = gradcam_map(model, xB, target_layer, infoB["target_idx"])
    imgB_224 = unnormalize_to_img01(xB[0])

    heatA = cam_to_heat_rgb(camA)
    heatB = cam_to_heat_rgb(camB)
    overlayA = np.clip(0.55 * imgA_224 + 0.45 * heatA, 0, 1)
    overlayB = np.clip(0.55 * imgB_224 + 0.45 * heatB, 0, 1)

    # --- Save side-by-side Grad-CAM image ---
    fig = plt.figure(figsize=(10, 4))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(overlayA)
    ax1.set_title(f"A: {imgs[0].name}\nTop-1: {nameA} ({infoA['target_prob']:.3f})")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(overlayB)
    ax2.set_title(f"B: {imgs[1].name}\nTop-1: {nameB} ({infoB['target_prob']:.3f})")
    ax2.axis("off")

    plt.tight_layout()
    grad_path = Path("outputs/compare/compare_gradcam.png")
    plt.savefig(grad_path, dpi=160)
    plt.close(fig)

    # --- Save side-by-side confidence trace plot ---
    cps = infoA["checkpoints"]
    xs = list(range(len(cps)))

    fig2 = plt.figure(figsize=(9, 4))
    plt.plot(xs, infoA["trace"], marker="o", label=f"A: {imgs[0].name}")
    plt.plot(xs, infoB["trace"], marker="o", label=f"B: {imgs[1].name}")
    plt.xticks(xs, cps)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Checkpoint (after block)")
    plt.ylabel("P(target class) with channel-gating")
    plt.title(f"Confidence trace (keep_ratio={keep_ratio})")
    plt.legend()
    plt.tight_layout()
    trace_path = Path("outputs/compare/compare_confidence_trace.png")
    plt.savefig(trace_path, dpi=160)
    plt.close(fig2)

    # --- Save summary ---
    summary = []
    summary.append(f"A file: {imgs[0].name}")
    summary.append(f"A top-1: {nameA} (id={infoA['target_idx']}) prob={infoA['target_prob']:.4f}")
    summary.append(f"A trace: {['%.4f' % v for v in infoA['trace']]}")
    summary.append("")
    summary.append(f"B file: {imgs[1].name}")
    summary.append(f"B top-1: {nameB} (id={infoB['target_idx']}) prob={infoB['target_prob']:.4f}")
    summary.append(f"B trace: {['%.4f' % v for v in infoB['trace']]}")
    summary.append("")
    summary.append(f"keep_ratio (channels): {keep_ratio}")

    Path("outputs/compare/compare_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"Saved: {grad_path}")
    print(f"Saved: {trace_path}")
    print("Saved: outputs/compare/compare_summary.txt")
    print("=== TWO IMAGE COMPARISON DONE ✅ ===")


if __name__ == "__main__":
    main()
