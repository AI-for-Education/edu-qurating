from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datasets import Dataset
from qr_scorer import QuratingScorer

qr_scorer = QuratingScorer()

app = FastAPI()


class ScoreBody(BaseModel):
    texts: list[str]
    model_type: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/score")
async def score(body: ScoreBody):
    if not body.texts:
        scores_json = {f"{key}_average": [] for key in qr_scorer.labels[body.model_type]}
    else:
        ds = Dataset.from_list([{"text": text} for text in body.texts])
        scores = qr_scorer._score_model(ds, model_type=body.model_type)
        scores_json = scores.to_dict()

    return JSONResponse(
        status_code=200,
        content={
            "scores": scores_json
        }
    )
