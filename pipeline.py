# pipeline.py
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------- utils ----------------
def run_step(cmd: list[str]):
    print("\n>>", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout.strip():
            print(result.stdout)
        if result.stderr.strip():
            print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    if result.stdout.strip():
        print(result.stdout)


def list_images(p: Path) -> list[Path]:
    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise SystemExit(f"Not an image file: {p}")
        return [p]
    if p.is_dir():
        imgs = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS]
        return sorted(imgs)
    raise SystemExit(f"Input path does not exist: {p}")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_jobs_root(jobs_root: Path):
    if jobs_root.exists():
        print(f"[CLEAN] Removing old jobs folder: {jobs_root}")
        shutil.rmtree(jobs_root)
    jobs_root.mkdir(parents=True, exist_ok=True)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="NN-XRAY pipeline (web-app ready)")
    ap.add_argument(
        "--input",
        default="inputs",
        help="Path to image OR folder with images",
    )
    ap.add_argument(
        "--job_id",
        default=None,
        help="Optional job id (default: timestamp)",
    )

    args = ap.parse_args()

    input_path = Path(args.input)
    images = list_images(input_path)
    if not images:
        raise SystemExit(f"No images found in: {input_path}")

    # --- JOB ROOT ---
    jobs_root = Path("data/jobs")
    clean_jobs_root(jobs_root)

    job_id = args.job_id or default_job_id()
    job_dir = jobs_root / job_id

    job_input_dir = ensure_dir(job_dir / "input")
    job_output_dir = ensure_dir(job_dir / "output")

    print(f"[JOB] id={job_id}")
    print(f"[JOB] root={job_dir}")

    # --- setup once ---
    run_step([sys.executable, "00_setup_env.py"])

    # --- per image ---
    for img in images:
        img = img.resolve()
        img_name = img.stem

        img_out = ensure_dir(job_output_dir / img_name)

        print("\n" + "=" * 60)
        print(f"PROCESSING IMAGE: {img.name}")
        print(f"OUTPUT DIR: {img_out}")
        print("=" * 60)

        # 02 preprocess
        run_step([
            sys.executable, "02_preprocess_io.py",
            "--image", str(img),
            "--out_dir", str(img_out),
        ])

        # 03 predict
        run_step([
            sys.executable, "03_predict.py",
            "--image", str(img),
            "--out_dir", str(img_out),
        ])

        # 04 feature maps
        run_step([
            sys.executable, "04_feature_maps.py",
            "--image", str(img),
            "--out_dir", str(img_out),
        ])

        # 05 activation stats
        run_step([
            sys.executable, "05_activation_stats.py",
            "--image", str(img),
            "--out_dir", str(img_out),
        ])

        # 06 gradcam
        run_step([
            sys.executable, "06_gradcam.py",
            "--image", str(img),
            "--out_dir", str(img_out),
        ])

        # 07 saliency
        run_step([
            sys.executable, "07_saliency.py",
            "--image", str(img),
            "--out_dir", str(img_out),
        ])

        # 08 confidence trace
        run_step([
            sys.executable, "08_confidence_trace.py",
            "--image", str(img),
            "--out_dir", str(img_out),
        ])

    # --- TWO IMAGE COMPARISON (only if exactly 2 images) ---
    if len(images) == 2:
        print("\n[COMPARE] Running two-image comparison")
        run_step([
            sys.executable, "09_two_image_comparison.py"
        ])
    else:
        print("\n[COMPARE] Skipped (need exactly 2 images)")

    print("\n" + "=" * 60)
    print("PIPELINE DONE")
    print(f"JOB DIR: {job_dir}")
    print("RESULTS:")
    print(f"  {job_dir}/output/<image_name>/")
    print("=" * 60)


if __name__ == "__main__":
    main()
