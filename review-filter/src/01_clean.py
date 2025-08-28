import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_FILE = Path("data/raw/reviews.csv")  # <-- change if your filename differs

KEEP_COLS = ["user_id","name","time","rating","text","pics","resp","gmap_id"]

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def has_text(v):
    try:
        return isinstance(v, str) and v.strip() != ""
    except Exception:
        return False

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Put your CSV at {RAW_FILE}. (Headers expected: {KEEP_COLS} [+ optional 'label'])")

    # Load
    df = pd.read_csv(RAW_FILE)

    # Keep known columns if present
    present = [c for c in KEEP_COLS if c in df.columns]
    df = df[present + ([c for c in df.columns if c not in present and c != "label"])]  # preserve others too
    if "label" in df.columns:
        df = df[[*present, "label"] + [c for c in df.columns if c not in present + ["label"]]]

    # Create id if missing
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    # Clean:
    # - Drop rows with empty/NaN text
    # - Drop rows with all-NaN *except* pics, resp (i.e., require some real data)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip() if "text" in df.columns else ""
    df = df[df["text"].apply(has_text)]  # require text
    # Drop rows where all non-exempt columns are NaN/blank
    exempt = set(["pics","resp"])
    cols_to_check = [c for c in df.columns if c not in exempt]
    df = df.dropna(subset=cols_to_check, how="all")

    # Save a fully-cleaned copy
    df.to_csv(OUT_DIR / "cleaned.csv", index=False)

    if "label" in df.columns:
        # Ensure labels are integers 0..3
        df = df.dropna(subset=["label"]).copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
        df = df[df["label"].isin([0,1,2,3])].copy()

        if df.empty:
            print("No valid labeled rows found. Wrote cleaned.csv only.")
            return

        # Stratified split 80/20
        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )

        # Keep just what training pipeline needs (id,text,label) + keep rating if present (useful later)
        base_cols = ["id","text","label"]
        if "rating" in df.columns:
            base_cols += ["rating"]
        train[base_cols].to_csv(OUT_DIR / "train.csv", index=False)
        test[base_cols].to_csv(OUT_DIR / "test.csv", index=False)

        print(f"Wrote train/test to {OUT_DIR} (rows: {len(train)}/{len(test)})")
    else:
        # No labels  prepare a small sample to annotate
        sample = df.sample(n=min(400, len(df)), random_state=42)[["id","text"]].copy()
        sample["label"] = ""  # annotate with 0..3
        sample.to_csv(OUT_DIR / "to_label.csv", index=False)
        df[["id","text"]].to_csv(OUT_DIR / "unlabeled.csv", index=False)
        print(f"No 'label' column found. Wrote unlabeled.csv and to_label.csv in {OUT_DIR}.")
        print("Annotate to_label.csv with labels: 0=valid,1=advertisement,2=irrelevant,3=rant_no_visit, then rerun this script.")
        return

if __name__ == "__main__":
    main()
