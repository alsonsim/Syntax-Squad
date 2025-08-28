import argparse, joblib, re, numpy as np

LABELS = {0:"valid", 1:"advertisement", 2:"irrelevant", 3:"rant_no_visit"}

ADS = re.compile(r"(http|www|promo|discount|use code|follow\s*@)", re.I)
NOV = re.compile(r"(never been|haven't been|didn't go|won't go|heard it(?:'s| is))", re.I)
IRR = re.compile(r"(my phone|ios|android|windows update|gpu driver)", re.I)

def rule_label(text: str):
    t = str(text)
    if ADS.search(t): return 1
    if NOV.search(t): return 3
    if IRR.search(t): return 2
    return None

NUM_ALL = 4

def expand_proba(p_row, classes):
    out = np.zeros(NUM_ALL, dtype=float)
    for i, cls in enumerate(classes):
        out[int(cls)] = p_row[i]
    if out.sum() == 0:
        out[:] = 1.0 / NUM_ALL
    return out

def onehot(label):
    v = np.zeros(NUM_ALL, dtype=float)
    if label is not None:
        v[int(label)] = 1.0
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--model", choices=["tfidf_lr","ensemble"], default="ensemble")
    args = ap.parse_args()

    clf = joblib.load("models/tfidf_lr/model.joblib")
    classes = clf.named_steps["clf"].classes_

    p_tfidf_raw = clf.predict_proba([args.text])[0]     # len = len(classes)
    p_tfidf = expand_proba(p_tfidf_raw, classes)        # len = 4

    if args.model == "tfidf_lr":
        pred = int(np.argmax(p_tfidf))
        print(pred)
        return

    lbl = rule_label(args.text)
    p_rules = onehot(lbl)
    p = 0.6 * p_tfidf + 0.4 * p_rules
    print(int(np.argmax(p)))

if __name__ == "__main__":
    main()
