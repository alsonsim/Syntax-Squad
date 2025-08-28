import glob, os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/reviews.csv")  # prefer this if present
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

TEXT_CANDS  = ["text","review_text","content","body","review","comment","reviewbody"]
LABEL_CANDS = ["label","labels","class","target","Label","Class","Target","'label'","\"label\""]

NAME_TO_ID = {
    "valid":0,
    "advertisement":1,"ad":1,"ads":1,"promo":1,"promotion":1,"spam":1,
    "irrelevant":2,"off-topic":2,"offtopic":2,"not relevant":2,"not_relevant":2,
    "rant_no_visit":3,"rant-no-visit":3,"rantnovisit":3,"no_visit_rant":3,"never_been":3,"never been":3,"heard":3,
}

def read_any():
    # choose file
    paths = [str(RAW)] if RAW.exists() else [*glob.glob("data/raw/*.csv"), *glob.glob("data/raw/*.xlsx")]
    if not paths:
        raise FileNotFoundError("Put your labeled file in data/raw/reviews.csv (CSV or XLSX).")
    p = paths[0]
    # read
    if p.lower().endswith(".xlsx"):
        df = pd.read_excel(p)
    else:
        df = None
        for enc in ["utf-8","utf-8-sig","utf-16","cp1252"]:
            for sep in [",",";","\t","|"]:
                try:
                    df = pd.read_csv(p, encoding=enc, sep=sep, engine="python")
                    if df is not None:
                        break
                except Exception:
                    continue
            if df is not None:
                break
        if df is None:
            raise RuntimeError(f"Failed to read {p}")
    return p, df

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        s = str(c).replace("\ufeff","").strip().strip("'").strip('"').lower().replace(" ", "_")
        cols.append(s)
    df = df.copy(); df.columns = cols
    return df

def find_text_col(df):
    for c in TEXT_CANDS:
        if c in df.columns: return c
    # fallback: longest avg length object col
    obj = [c for c in df.columns if df[c].dtype == "object"]
    if not obj: raise ValueError("No text-like column found.")
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())

def coerce_labels(df, label_col):
    s = df[label_col]
    if s.dtype == "object":
        s = s.astype(str).str.strip().str.lower().map(NAME_TO_ID)
    else:
        s = pd.to_numeric(s, errors="coerce")
    s = s.astype("Int64")
    df[label_col] = s
    return df

def main():
    p, df = read_any()
    df = normalize_headers(df)

    # ensure id
    if "id" not in df.columns or df["id"].duplicated().any():
        df.insert(0, "id", range(1, len(df)+1))

    # text
    text_col = find_text_col(df)
    df[text_col] = df[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df[text_col].str.len() > 0]

    # label (optional)
    label_col = None
    for cand in LABEL_CANDS:
        if cand in df.columns:
            label_col = cand if cand == "label" else "label"
            if cand != "label":
                df.rename(columns={cand: "label"}, inplace=True)
            break

    if label_col is not None:
        df = coerce_labels(df, "label")

    # save cleaned
    df.to_csv(OUT/"cleaned.csv", index=False, encoding="utf-8")

    # train/test if labels valid
    if "label" in df.columns and df["label"].notna().any():
        df_l = df[df["label"].isin([0,1,2,3])].copy()
        if len(df_l) >= 20 and df_l["label"].nunique() > 1:
            keep = ["id", text_col, "label"] + (["rating"] if "rating" in df_l.columns else [])
            tr, te = train_test_split(df_l[keep], test_size=0.2, random_state=42, stratify=df_l["label"])
            tr.to_csv(OUT/"train.csv", index=False, encoding="utf-8")
            te.to_csv(OUT/"test.csv",  index=False, encoding="utf-8")
            print(f"Wrote train/test to {OUT} using text='{text_col}', label='label'.")
            return

    # else prepare unlabeled / to_label
    unl = df[["id", text_col]].copy()
    unl.to_csv(OUT/"unlabeled.csv", index=False, encoding="utf-8")
    samp = unl.sample(n=min(400, len(unl)), random_state=42).copy()
    samp["label"] = ""
    samp.to_csv(OUT/"to_label.csv", index=False, encoding="utf-8")
    print(f"No (sufficient) labels. Wrote unlabeled.csv and to_label.csv (text column='{text_col}').")

if __name__ == "__main__":
    main()
