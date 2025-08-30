import argparse, re, numpy as np, joblib, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {0:"valid",1:"advertisement",2:"irrelevant",3:"rant_no_visit"}

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

def expand_tfidf(p_row, classes, k=4):
    out = np.zeros(k, dtype=float)
    for i, cls in enumerate(classes): out[int(cls)] = p_row[i]
    if out.sum()==0: out[:] = 1.0/k
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    text = args.text

    # TF-IDF
    tfidf = joblib.load("models/tfidf_lr/model.joblib")
    tfidf_classes = tfidf.named_steps["clf"].classes_
    p_tfidf = expand_tfidf(tfidf.predict_proba([text])[0], tfidf_classes)

    # DistilBERT
    tok = AutoTokenizer.from_pretrained("models/distilbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("models/distilbert")
    mdl.eval()
    with torch.no_grad():
        enc = tok([text], padding=True, truncation=True, return_tensors="pt")
        p_bert = torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()[0]

    # Rules
    p_rules = onehot(rule_id(text))

    # Blend
    w_bert, w_tfidf, w_rules = 0.5, 0.3, 0.2
    p = w_bert*p_bert + w_tfidf*p_tfidf + w_rules*p_rules
    k = int(np.argmax(p))
    print({"label_id": k, "label_name": LABELS[k], "confidence": float(p[k])})

if __name__ == "__main__":
    main()
