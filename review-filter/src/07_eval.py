from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

PREDS_DIR = Path("outputs/preds")
METRICS_DIR = Path("outputs/metrics"); METRICS_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for fp in PREDS_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(fp)
        if not {"label","pred"}.issubset(df.columns):
            continue
        rep = classification_report(df["label"], df["pred"], output_dict=True, zero_division=0)
        rows.append({
            "file": fp.name,
            "accuracy": rep.get("accuracy", 0.0),
            "macro_f1": rep.get("macro avg", {}).get("f1-score", 0.0),
            "f1_valid": rep.get("0", {}).get("f1-score", 0.0),
            "f1_ad": rep.get("1", {}).get("f1-score", 0.0),
            "f1_irrelevant": rep.get("2", {}).get("f1-score", 0.0),
            "f1_rant": rep.get("3", {}).get("f1-score", 0.0),
            "support": int(rep.get("macro avg", {}).get("support", 0))
        })
    except Exception as e:
        print(f"[warn] skipping {fp}: {e}")

if rows:
    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(summary.to_string(index=False))
    summary.to_csv(METRICS_DIR/"summary.csv", index=False)
    print("\nSaved ->", METRICS_DIR/"summary.csv")
else:
    print("No suitable preds found in outputs/preds (need columns: label, pred).")
