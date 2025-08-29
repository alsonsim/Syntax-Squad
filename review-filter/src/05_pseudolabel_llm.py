from pathlib import Path
import re, numpy as np, pandas as pd, joblib

PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)
OUT_PSEUDO = PROC/"pseudo_train.csv"
OUT_MERGED = PROC/"train_plus_pseudo.csv"
TFIDF_MODEL = Path("models/tfidf_lr/model.joblib")

# 1) load unlabeled and optional labeled train
unl = pd.read_csv(PROC/"unlabeled.csv")
base_train = pd.read_csv(PROC/"train.csv") if (PROC/"train.csv").exists() else None

# 2) rules
ADS = re.compile(r"(http|www|promo|discount|use code|follow\s*@)", re.I)
NOV = re.compile(r"(never been|haven't been|didn't go|won't go|heard it(?:'s| is))", re.I)
IRR = re.compile(r"(my phone|ios|android|windows update|gpu driver)", re.I)
def rule_label(t):
    t = str(t or "")
    if ADS.search(t): return 1
    if NOV.search(t): return 3
    if IRR.search(t): return 2
    return None

# 3) tfidf probs (expand to 4 classes if needed)
clf = joblib.load(TFIDF_MODEL)
classes = clf.named_steps["clf"].classes_

def expand_proba(p_row):
    out = np.zeros(4, dtype=float)
    for i, cls in enumerate(classes):
        out[int(cls)] = p_row[i]
    if out.sum() == 0: out[:] = 1.0/4
    return out

P = clf.predict_proba(unl["text"].astype(str))
P = np.vstack([expand_proba(p) for p in P])

# 4) blend with rules (0.6 tfidf / 0.4 rules)
R = np.zeros_like(P)
for i, t in enumerate(unl["text"].astype(str)):
    lbl = rule_label(t)
    if lbl is not None: R[i, int(lbl)] = 1.0

proba = 0.6*P + 0.4*R
pred  = proba.argmax(axis=1)
conf  = proba.max(axis=1)

# 5) keep only high confidence (you can tweak threshold)
THRESH = 0.80
mask = conf >= THRESH
pseudo = unl.loc[mask, ["id","text"]].copy()
pseudo["label"] = pred[mask]
pseudo["confidence"] = conf[mask]

pseudo.to_csv(OUT_PSEUDO, index=False)
print(f"[pseudolabel] wrote {len(pseudo)} rows -> {OUT_PSEUDO} (threshold={THRESH})")

# 6) optionally merge with labeled train to create a larger set
if base_train is not None and len(pseudo) > 0:
    merged = pd.concat([base_train[["id","text","label"]], pseudo[["id","text","label"]]], ignore_index=True)
    merged.to_csv(OUT_MERGED, index=False)
    print(f"[pseudolabel] merged train+pseudo -> {OUT_MERGED}")
