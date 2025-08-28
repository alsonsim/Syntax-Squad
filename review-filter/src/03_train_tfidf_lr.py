from pathlib import Path
import os, json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

processed = Path("data/processed")
outputs   = Path("outputs")
models    = Path("models/tfidf_lr")
(outputs / "preds").mkdir(parents=True, exist_ok=True)
(outputs / "metrics").mkdir(parents=True, exist_ok=True)
models.mkdir(parents=True, exist_ok=True)

train_csv = processed / "train.csv"
test_csv  = processed / "test.csv"
if not train_csv.exists():
    raise FileNotFoundError("Missing data/processed/train.csv. Run 01_clean.py first.")

train = pd.read_csv(train_csv)
test  = pd.read_csv(test_csv) if test_csv.exists() else train.copy()

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipe.fit(train["text"].astype(str), train["label"])

preds = pipe.predict(test["text"].astype(str))
report = classification_report(test["label"], preds, output_dict=True, zero_division=0)

pd.DataFrame({"id": test.get("id", pd.Series(range(len(test)))), 
              "text": test["text"], 
              "label": test["label"], 
              "pred": preds}).to_csv(outputs/"preds"/"tfidf_lr_test.csv", index=False)

(Path(outputs/"metrics"/"tfidf_lr_test.json")).write_text(json.dumps(report, indent=2))

joblib.dump(pipe, models/"model.joblib")
print(f"[tfidf_lr] saved model -> {models/'model.joblib'}")
print(f"[tfidf_lr] wrote metrics -> {outputs/'metrics'/'tfidf_lr_test.json'}")
