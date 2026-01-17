import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess, weights


@torch.no_grad()
def predict_topk(model, preprocess, img_path: Path, device: torch.device, k: int = 5):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # (1,3,224,224)

    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]  # (1000,)

    top_probs, top_idxs = torch.topk(probs, k)
    return top_idxs.cpu().tolist(), top_probs.cpu().tolist()


def main():
    print("=== PREDICT CHECK ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out_dir", required=True, help="Per-image output directory")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"[ERROR] Image not found: {img_path}")

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, preprocess, weights = load_model(device)

    labels = weights.meta.get("categories", None)

    top_idxs, top_probs = predict_topk(model, preprocess, img_path, device, k=5)

    results = []
    for idx, prob in zip(top_idxs, top_probs):
        name = labels[idx] if labels else f"class_{idx}"
        results.append({"class_id": idx, "class_name": name, "prob": float(prob)})

    out = {
        "image": str(img_path),
        "device": str(device),
        "top5": results,
    }

    # save json
    json_path = pred_dir / "prediction.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # save txt
    txt_path = pred_dir / "prediction.txt"
    lines = [f"Image: {img_path}", f"Device: {device}", "Top-5:"]
    for r in results:
        lines.append(f"- {r['class_name']} (id={r['class_id']}): {r['prob']:.4f}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print("Top-5:")
    for r in results:
        print(f"- {r['class_name']} (id={r['class_id']}): {r['prob']:.4f}")

    print("=== PREDICT DONE ===")


if __name__ == "__main__":
    main()
