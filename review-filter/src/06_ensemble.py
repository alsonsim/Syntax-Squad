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

# get class ids the model actually learned (e.g., array([0,2]))
classes = clf.named_steps["clf"].classes_
NUM_ALL = 4

def expand_proba(p_row, classes):
    out = np.zeros(NUM_ALL, dtype=float)
    for i, cls in enumerate(classes):
        out[int(cls)] = p_row[i]
    # if no prob mass assigned (edge case), fallback to uniform
    if out.sum() == 0:
        out[:] = 1.0 / NUM_ALL
    return out

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

def onehot(label):
    v = np.zeros(NUM_ALL, dtype=float)
    if label is not None:
        v[int(label)] = 1.0
    return v

# predict
texts = df["text"].astype(str).tolist()
proba_raw = clf.predict_proba(texts)          # shape [n, len(classes)]
proba_tf  = np.vstack([expand_proba(p, classes) for p in proba_raw])  # [n,4]
proba_ru  = np.vstack([onehot(rule_label(t)) for t in texts])         # [n,4]

# soft vote
w_rules, w_tfidf = 0.4, 0.6
proba = w_tfidf * proba_tf + w_rules * proba_ru
pred  = proba.argmax(axis=1)

# save outputs
out_preds = outputs/"preds"/"ensemble_test.csv"
pd.DataFrame({
    "id": df.get("id", pd.Series(range(len(df)))),
    "text": df["text"],
    "label": df.get("label"),
    "pred": pred
}).to_csv(out_preds, index=False)

# metrics if labels available
if "label" in df.columns:
    report = classification_report(df["label"], pred, output_dict=True, zero_division=0)
    (outputs/"metrics"/"ensemble_test.json").write_text(json.dumps(report, indent=2))
    print(f"[ensemble] macro-F1: {report['macro avg']['f1-score']:.3f}")
else:
    print("[ensemble] no labels found; wrote predictions only.")

print(f"[ensemble] wrote predictions -> {out_preds}")
