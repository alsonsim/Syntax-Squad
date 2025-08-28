import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import re
from sklearn.metrics import classification_report

# paths
processed = Path("data/processed")
outputs   = Path("outputs")
models    = Path("models/tfidf_lr")
(outputs / "preds").mkdir(parents=True, exist_ok=True)
(outputs / "metrics").mkdir(parents=True, exist_ok=True)

# load data
test_csv = processed/"test.csv"
if not test_csv.exists():
    raise FileNotFoundError("Missing data/processed/test.csv. Run 01_clean.py first.")
df = pd.read_csv(test_csv)

# load model
model_path = models/"model.joblib"
if not model_path.exists():
    raise FileNotFoundError("Missing TF-IDF model. Run 03_train_tfidf_lr.py first.")
clf = joblib.load(model_path)

# rules (return class id or None)
ADS_PAT       = re.compile(r"(http|www|promo|discount|use code|follow\s*@)", re.I)
NO_VISIT_PAT  = re.compile(r"(never been|haven't been|didn't go|won't go|heard it(?:'s| is))", re.I)
IRREL_PAT     = re.compile(r"(my phone|ios|android|windows update|gpu driver)", re.I)

def rule_label(text: str):
    t = str(text)
    if ADS_PAT.search(t):       return 1
    if NO_VISIT_PAT.search(t):  return 3
    if IRREL_PAT.search(t):     return 2
    return None

NUM_CLASSES = 4

def onehot(label, k=NUM_CLASSES):
    v = np.zeros(k, dtype=float)
    if label is not None:
        v[int(label)] = 1.0
    return v

# predict
texts = df["text"].astype(str).tolist()
proba_tfidf = clf.predict_proba(texts)  # shape [n,4]
proba_rules = np.vstack([onehot(rule_label(t)) for t in texts])

# soft vote; tweak weights if you like
w_rules, w_tfidf = 0.4, 0.6
proba = w_tfidf * proba_tfidf + w_rules * proba_rules
pred  = proba.argmax(axis=1)

# save outputs
out_preds = outputs/"preds"/"ensemble_test.csv"
pd.DataFrame({
    "id": df.get("id", pd.Series(range(len(df)))),
    "text": df["text"],
    "label": df["label"] if "label" in df else pd.Series([None]*len(df)),
    "pred": pred
}).to_csv(out_preds, index=False)

# metrics if labels available
metrics = {}
if "label" in df.columns:
    report = classification_report(df["label"], pred, output_dict=True, zero_division=0)
    metrics = report
    (outputs/"metrics"/"ensemble_test.json").write_text(json.dumps(report, indent=2))

print(f"[ensemble] wrote predictions -> {out_preds}")
if metrics:
    print(f"[ensemble] macro-F1: {metrics['macro avg']['f1-score']:.3f}")
