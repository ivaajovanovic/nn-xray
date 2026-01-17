from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

from api.schemas import InferenceResponse


app = FastAPI(title="NN-XRAY API")

# Folders
JOBS_ROOT = Path("data/jobs")
INPUTS_ROOT = Path("inputs")

JOBS_ROOT.mkdir(parents=True, exist_ok=True)
INPUTS_ROOT.mkdir(parents=True, exist_ok=True)


@app.post("/infer", response_model=InferenceResponse)
async def infer(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Create job
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded image
    img_path = INPUTS_ROOT / f"{job_id}_{file.filename}"
    with img_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run pipeline
    try:
        subprocess.run(
            [
                sys.executable,
                "pipeline.py",
                "--input",
                str(img_path),
                "--job_dir",
                str(job_dir),
                "--all",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail="Pipeline execution failed")

    return InferenceResponse(
        job_id=job_id,
        status="done",
        results_path=str(job_dir / "output"),
    )
