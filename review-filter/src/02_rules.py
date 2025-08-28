import re, json
from pathlib import Path
import pandas as pd

processed = Path("data/processed")
outputs   = Path("outputs")
(outputs / "preds").mkdir(parents=True, exist_ok=True)
(outputs / "metrics").mkdir(parents=True, exist_ok=True)

# Try test first; fall back to train if test doesn't exist
csv_path = processed / "test.csv"
if not csv_path.exists():
    csv_path = processed / "train.csv"

df = pd.read_csv(csv_path)

ADS_PAT       = re.compile(r"(http|www|promo|discount|use code|follow\s*@)", re.I)
NO_VISIT_PAT  = re.compile(r"(never been|haven't been|didn't go|won't go|heard it(?:'s| is))", re.I)
IRREL_PAT     = re.compile(r"(my phone|ios|android|windows update|gpu driver)", re.I)

def predict_rule(text: str) -> int:
    t = str(text)
    if ADS_PAT.search(t):      return 1  # advertisement
    if NO_VISIT_PAT.search(t): return 3  # rant_no_visit
    if IRREL_PAT.search(t):    return 2  # irrelevant
    return 0                   # valid

df["pred"] = df["text"].astype(str).apply(predict_rule)

metrics = {}
if "label" in df.columns:
    acc = float((df["pred"] == df["label"]).mean())
    metrics["accuracy"] = acc

out_preds = outputs / "preds" / "rules.csv"
df.to_csv(out_preds, index=False)

out_metrics = outputs / "metrics" / "rules.json"
Path(out_metrics).write_text(json.dumps(metrics, indent=2))

print(f"[rules] wrote predictions -> {out_preds}")
if metrics:
    print(f"[rules] accuracy: {metrics['accuracy']:.3f}")
