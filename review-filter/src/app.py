from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import numpy as np

from src.demo_infer import rule_id, LABELS, expand_proba, onehot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join("models", "tfidf_lr", "model.joblib")
clf = joblib.load(MODEL_PATH)
classes = clf.named_steps["clf"].classes_
NUM_ALL = len(LABELS)

class ReviewRequest(BaseModel):
    text: str

@app.post("/predict/")
async def predict(request: ReviewRequest):
    text = request.text

    # 1. Rules
    rule_lbl = rule_id(text)
    p_rules = onehot(rule_lbl)
    
    # 2. Model proba
    p_tfidf_raw = clf.predict_proba([text])[0]
    p_tfidf = expand_proba(p_tfidf_raw, classes)
    
    # 3. Weighted ensemble (reuse your CLI logic)
    p_final = 0.6 * p_tfidf + 0.4 * p_rules
    pred_final = int(np.argmax(p_final))

    # Optionally, output model and rule results for debugging
    # print(f"text: {text}, rule: {rule_lbl}, model: {p_tfidf}, final: {p_final}, pred_final: {pred_final}")
    
    return {
        "label_id": pred_final,
        "label": LABELS.get(pred_final, "UNKNOWN")
    }
