import pandas as pd

p = "data/raw/reviews.csv"
df = pd.read_csv(p, engine="python")

# strip quotes/spaces, lowercase, underscores
new_cols = []
for c in df.columns:
    c2 = str(c).strip().strip("'").strip('"').lower().replace(" ", "_")
    new_cols.append(c2)
df.columns = new_cols

# coerce label to integer ids if present
if "label" in df.columns:
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

df.to_csv(p, index=False, encoding="utf-8")
print("Normalized columns:", list(df.columns))
print("Label uniques:", df["label"].dropna().unique()[:20] if "label" in df.columns else "no label")
