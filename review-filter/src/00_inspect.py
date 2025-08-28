import os, glob, pandas as pd

paths = [
  "data/raw/reviews.csv",
  *glob.glob("data/raw/*.csv"),
  *glob.glob("data/raw/*.xlsx"),
]
paths = [p for p in paths if os.path.exists(p)]
print("CANDIDATE FILES:", paths)

def read_any(p):
    if p.lower().endswith(".xlsx"):
        return pd.read_excel(p)
    # try encodings / separators
    for enc in ["utf-8", "utf-8-sig", "utf-16", "cp1252"]:
        for sep in [",", ";", "\t"]:
            try:
                return pd.read_csv(p, sep=sep, encoding=enc, engine="python")
            except Exception:
                pass
    raise RuntimeError(f"Failed to read {p}")

if not paths:
    raise SystemExit("No files found in data/raw/. Put your labeled file there as reviews.csv")

p = paths[0]
df = read_any(p)
print("USING FILE:", p)
print("COLUMNS RAW:", [repr(c) for c in df.columns.tolist()][:20])
print(df.head(3))
print("Dtypes:", df.dtypes.to_dict())
if "label" in df.columns: print("Label unique:", df["label"].dropna().unique()[:20])
