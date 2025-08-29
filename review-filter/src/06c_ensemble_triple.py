import json, re
from pathlib import Path
import numpy as np, pandas as pd, joblib, torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROC = Path("data/processed")
OUT  = Path("outputs"); (OUT/"preds").mkdir(parents=True, exist_ok=True); (OUT/"metrics").mkdir(parents=True, exist_ok=True)

test_csv = PROC/"test.csv"
if not test_csv.exists(): raise FileNotFoundError("Run 01_clean.py to create data/processed/test.csv.")
df = pd.read_csv(test_csv)

# --- Rules ---
ADS = re.compile(r"(http|www|promo|discount|use code|follow\s*@)", re.I)
NOV = re.compile(r"(never been|haven't been|didn't go|won't go|heard it(?:'s| is))", re.I)
IRR = re.compile(r"(my phone|ios|android|windows update|gpu driver)", re.I)
def rule_id(t):
    t=str(t)
    if ADS.search(t): return 1
    if NOV.search(t): return 3
    if IRR.search(t): return 2
    return None

def onehot(lbl, k=4):
    v = np.zeros(k, dtype=float)
    if lbl is not None: v[int(lbl)] = 1.0
    return v

# --- TF-IDF ---
tfidf_path = Path("models/tfidf_lr/model.joblib")
if not tfidf_path.exists(): raise FileNotFoundError("Train TF-IDF first (03_train_tfidf_lr.py).")
tfidf = joblib.load(tfidf_path)
tfidf_classes = tfidf.named_steps["clf"].classes_
def expand_tfidf(p_row, classes, k=4):
    out = np.zeros(k, dtype=float)
    for i, cls in enumerate(classes): out[int(cls)] = p_row[i]
    if out.sum()==0: out[:] = 1.0/k
    return out

texts = df["text"].astype(str).tolist()
P_tfidf_raw = tfidf.predict_proba(texts)
P_tfidf = np.vstack([expand_tfidf(p, tfidf_classes) for p in P_tfidf_raw])

# --- DistilBERT ---
bert_dir = Path("models/distilbert")
if not bert_dir.exists(): raise FileNotFoundError("Train DistilBERT first (04_train_distilbert.py).")
tok = AutoTokenizer.from_pretrained(str(bert_dir))
mdl = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
mdl.eval()

BATCH=32
P_bert = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt")
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        P_bert.append(probs)
P_bert = np.vstack(P_bert)

# --- Rules matrix ---
P_rules = np.vstack([onehot(rule_id(t)) for t in texts])

# --- Blend (weights are tunable) ---
w_bert, w_tfidf, w_rules = 0.5, 0.3, 0.2
P = w_bert*P_bert + w_tfidf*P_tfidf + w_rules*P_rules
pred = P.argmax(axis=1)

out_preds = OUT/"preds"/"triple_ensemble_test.csv"
pd.DataFrame({"id": df.get("id", pd.Series(range(len(df)))), "text": df["text"], "label": df.get("label"), "pred": pred}).to_csv(out_preds, index=False)

if "label" in df.columns:
    rep = classification_report(df["label"], pred, output_dict=True, zero_division=0)
    (OUT/"metrics"/"triple_ensemble_test.json").write_text(json.dumps(rep, indent=2))
    print(f"[triple] macro-F1: {rep['macro avg']['f1-score']:.3f}")

print(f"[triple] wrote predictions -> {out_preds}")
