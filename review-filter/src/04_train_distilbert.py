from pathlib import Path
import json, numpy as np, pandas as pd, torch
from sklearn.metrics import classification_report
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)

PROC = Path("data/processed")
OUT  = Path("outputs"); (OUT/"metrics").mkdir(parents=True, exist_ok=True); (OUT/"preds").mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models/distilbert"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_csv = PROC/"train.csv"
test_csv  = PROC/"test.csv"
if not train_csv.exists(): raise FileNotFoundError("Missing data/processed/train.csv. Run 01_clean.py first.")

train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv) if test_csv.exists() else pd.read_csv(train_csv)

# ensure labels are 0..3
train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").astype("Int64")
test_df["label"]  = pd.to_numeric(test_df["label"], errors="coerce").astype("Int64")

MODEL_NAME = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize(batch): return tok(batch["text"], truncation=True)

train_ds = Dataset.from_pandas(train_df[["text","label"]], preserve_index=False)
eval_ds  = Dataset.from_pandas(test_df[["text","label"]], preserve_index=False)
ds_tok_train = train_ds.map(tokenize, batched=True, remove_columns=["text"])
ds_tok_eval  = eval_ds.map(tokenize, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tokenizer=tok)

# class weights (optional, helps imbalance)
counts = train_df["label"].value_counts().reindex([0,1,2,3]).fillna(0).astype(int)
tot = counts.sum()
weights = torch.tensor([(tot/(c if c>0 else 1)) for c in counts], dtype=torch.float)
weights = weights / weights.mean()

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # kwargs absorbs num_items_in_batch (and any future extras)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Keep args minimal for older transformers
args = TrainingArguments(
    output_dir=str(MODEL_DIR/"runs"),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=ds_tok_train,
    eval_dataset=ds_tok_eval,
    tokenizer=tok,
    data_collator=collator,
)

trainer.train()

# Manual evaluation
pred = trainer.predict(ds_tok_eval)
logits = pred.predictions
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
pred_labels = probs.argmax(axis=1)

rep = classification_report(test_df["label"].values, pred_labels, output_dict=True, zero_division=0)
(Path(OUT/"metrics"/"distilbert_val.json")).write_text(json.dumps(rep, indent=2))

pred_df = test_df.copy()
pred_df["pred"] = pred_labels
pred_df.to_csv(OUT/"preds"/"distilbert_val.csv", index=False)

# Save model + tokenizer
trainer.save_model(str(MODEL_DIR))
tok.save_pretrained(str(MODEL_DIR))

print("[distilbert] saved model ->", MODEL_DIR)
print("[distilbert] metrics ->", OUT/"metrics"/"distilbert_val.json")
print("[distilbert] preds ->", OUT/"preds"/"distilbert_val.csv")
