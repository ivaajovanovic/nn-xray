from pydantic import BaseModel
from typing import List, Optional


class PredictionItem(BaseModel):
    class_id: int
    class_name: str
    prob: float


class InferenceResponse(BaseModel):
    job_id: str
    status: str
    results_path: str

    top1: Optional[PredictionItem] = None
    top5: Optional[List[PredictionItem]] = None
