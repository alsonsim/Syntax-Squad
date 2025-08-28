import argparse, joblib, re, numpy as np

# rules
ADS_PAT       = re.compile(r"(http|www|promo|discount|use code|follow\s*@)", re.I)
NO_VISIT_PAT  = re.compile(r"(never been|haven't been|didn't go|won't go|heard it(?:'s| is))", re.I)
IRREL_PAT     = re.compile(r"(my phone|ios|android|windows update|gpu driver)", re.I)

def rule_label(text: str):
    t = str(text)
    if ADS_PAT.search(t):       return 1
    if NO_VISIT_PAT.search(t):  return 3
    if IRREL_PAT.search(t):     return 2
    return None

NUM_CLASSES = 4

def onehot(label):
    v = np.zeros(NUM_CLASSES, dtype=float)
    if label is not None:
        v[int(label)] = 1.0
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ensemble", choices=["tfidf_lr","ensemble"])
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    # load tfidf model
    clf = joblib.load("models/tfidf_lr/model.joblib")

    # tfidf proba
    p_tfidf = clf.predict_proba([args.text])[0]

    if args.model == "tfidf_lr":
        pred = int(np.argmax(p_tfidf))
        print(pred)
        return

    # ensemble
    lbl = rule_label(args.text)
    p_rules = onehot(lbl)
    p = 0.6*p_tfidf + 0.4*p_rules
    print(int(np.argmax(p)))

if __name__ == "__main__":
    main()
