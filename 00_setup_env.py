import os
import sys
import torch
from datetime import datetime


def main():
    print("=== SETUP CHECK ===")

    # Python
    print(f"Python version: {sys.version}")

    # PyTorch
    print(f"PyTorch version: {torch.__version__}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU")

    # ✅ FIX: definisana lista foldera
    folders = [
        "outputs",
        "outputs/input",
        "outputs/predictions",
        "outputs/features",
        "outputs/heatmaps",
        "outputs/trace",
        "outputs/debug",
        "data/jobs",
    ]

    for f in folders:
        os.makedirs(f, exist_ok=True)

    # Sanity file
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sanity_path = "outputs/debug/setup_ok.txt"
    with open(sanity_path, "w", encoding="utf-8") as f:
        f.write("SETUP OK\n")
        f.write(f"Time: {now}\n")
        f.write(f"Device: {device}\n")
        f.write(f"PyTorch: {torch.__version__}\n")

    print("Folders created")
    print(f"Sanity file written: {sanity_path}")
    print("=== SETUP DONE ===")


if __name__ == "__main__":
    main()
