import os
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
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


# ----------------- gating (makes trace non-flat) -----------------
def gate_topk_channels(z, keep_ratio=0.25):
    """
    Keep only top-k channels by mean(|activation|) across spatial dims.
    z: (1,C,H,W)
    """
    assert z.dim() == 4
    C = z.shape[1]
    k = max(1, int(C * keep_ratio))
    score = z.abs().mean(dim=(0, 2, 3))          # (C,)
    topk = torch.topk(score, k).indices          # (k,)
    mask = torch.zeros(C, device=z.device)
    mask[topk] = 1.0
    return z * mask.view(1, C, 1, 1)


def gate_topk_spatial(z, keep_ratio=0.15):
    """
    Keep only top-k spatial positions by mean(|activation|) across channels.
    z: (1,C,H,W)
    """
    assert z.dim() == 4
    H, W = z.shape[2], z.shape[3]
    k = max(1, int(H * W * keep_ratio))
    score_hw = z.abs().mean(dim=1, keepdim=True)     # (1,1,H,W)
    flat = score_hw.view(1, 1, -1)                   # (1,1,HW)
    topk = torch.topk(flat, k, dim=-1).indices       # (1,1,k)
    mask = torch.zeros_like(flat)
    mask.scatter_(-1, topk, 1.0)
    mask = mask.view(1, 1, H, W)                     # (1,1,H,W)
    return z * mask


# ----------------- forward split for ResNet18 -----------------
@torch.no_grad()
def forward_checkpoints(model, x):
    """
    Returns activations AFTER each block:
    stem, layer1, layer2, layer3, layer4
    """
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
    """
    Continue the network from checkpoint tensor z.
    start_cp is one of: stem/layer1/layer2/layer3/layer4
    """
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


# ----------------- main -----------------
def main():
    print("=== CONFIDENCE TRACE (GATED) CHECK ===")
    os.makedirs("outputs/trace", exist_ok=True)

    img_path = Path("input.jpg")
    if not img_path.exists():
        print("[ERROR] Ne postoji input.jpg u folderu projekta.")
        return

    device = get_device()
    model, preprocess, labels = load_model(device)

    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    # Final target = final top-1
    probs_final = predict_probs(model, x)
    target_idx = int(torch.argmax(probs_final).item())
    target_prob = float(probs_final[target_idx].item())
    target_name = labels[target_idx] if labels else f"class_{target_idx}"
    print(f"Target (final top-1): {target_name} (id={target_idx}) prob={target_prob:.4f}")

    # 1) capture checkpoint tensors
    acts = forward_checkpoints(model, x)
    checkpoints = ["stem", "layer1", "layer2", "layer3", "layer4", "final"]

    # 2) compute traces
    keep_ch_ratio = 0.25   # 25% top channels (dobro za demo)
    keep_sp_ratio = 0.15   # 15% top spatial positions

    rows = []
    for cp in checkpoints:
        if cp == "final":
            p_raw = float(probs_final[target_idx].item())
            p_ch  = p_raw
            p_sp  = p_raw
        else:
            z = acts[cp]

            # RAW (kontrola)
            probs_raw = forward_from_checkpoint(model, z, cp)
            p_raw = float(probs_raw[target_idx].item())

            # GATE channels (ovo pravi "trace" koji se menja)
            z_ch = gate_topk_channels(z, keep_ratio=keep_ch_ratio)
            probs_ch = forward_from_checkpoint(model, z_ch, cp)
            p_ch = float(probs_ch[target_idx].item())

            # GATE spatial (opciono)
            z_sp = gate_topk_spatial(z, keep_ratio=keep_sp_ratio)
            probs_sp = forward_from_checkpoint(model, z_sp, cp)
            p_sp = float(probs_sp[target_idx].item())

        rows.append({
            "checkpoint": cp,
            "p_raw": p_raw,
            "p_gate_channels": p_ch,
            "p_gate_spatial": p_sp
        })

    # Save CSV
    csv_path = Path("outputs/trace/confidence_trace.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "p_raw", "p_gate_channels", "p_gate_spatial"])
        w.writeheader()
        w.writerows(rows)

    # Save TXT
    txt_path = Path("outputs/trace/confidence_trace.txt")
    lines = [
        f"Image: {img_path}",
        f"Device: {device}",
        f"Target (final top-1): {target_name} (id={target_idx})",
        f"Gate: channels keep_ratio={keep_ch_ratio}, spatial keep_ratio={keep_sp_ratio}",
        ""
    ]
    for r in rows:
        lines.append(
            f"{r['checkpoint']}: raw={r['p_raw']:.6f} | gate_channels={r['p_gate_channels']:.6f} | gate_spatial={r['p_gate_spatial']:.6f}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # Plot
    plot_path = Path("outputs/trace/confidence_trace.png")
    xs = list(range(len(rows)))
    xlabels = [r["checkpoint"] for r in rows]
    y_raw = [r["p_raw"] for r in rows]
    y_ch  = [r["p_gate_channels"] for r in rows]
    y_sp  = [r["p_gate_spatial"] for r in rows]

    plt.figure(figsize=(9, 4))
    plt.plot(xs, y_raw, marker="o", label="raw (control)")
    plt.plot(xs, y_ch, marker="o", label="gate_channels (trace)")
    plt.plot(xs, y_sp, marker="o", label="gate_spatial (optional)")
    plt.xticks(xs, xlabels)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Checkpoint (measured after block)")
    plt.ylabel("P(target class)")
    plt.title(f"Confidence trace for target: {target_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {plot_path}")
    print("=== CONFIDENCE TRACE DONE ✅ ===")


if __name__ == "__main__":
    main()
