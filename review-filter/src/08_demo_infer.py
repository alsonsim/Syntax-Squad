import argparse, joblib, re
import numpy as np


LABELS = {0:"valid", 1:"advertisement", 2:"irrelevant", 3:"rant_no_visit"}


ADS = re.compile(r"(http|www|promo|discount|use code|follow\s*@)", re.I)
# Expanded 'no visit' pattern to include more phrases
NOV = re.compile(
    r"(never been|haven't been|didn't go|won't go|heard it(?:'s| is)|"
    r"don't\s+know|don't\s+even\s+know|no\s+idea\s+about|not\s+familiar\s+with|"
    r"haven't\s+tried|haven't\s+eaten\s+at|not\s+been\s+to)",
    re.I
)
IRR = re.compile(r"(my phone|ios|android|windows update|gpu driver)", re.I)


def rule_id(t):
    t = str(t)
    if ADS.search(t):
        print("Rule label: 1 (advertisement)")
        return 1
    if NOV.search(t):
        print("Rule label: 3 (rant_no_visit)")
        return 3
    if IRR.search(t):
        print("Rule label: 2 (irrelevant)")
        return 2
    print("Rule label: None (unassigned)")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", choices=["tfidf_lr","ensemble"], default="ensemble")
    args = parser.parse_args()

    clf = joblib.load("models/tfidf_lr/model.joblib")
    # Assuming pipeline has 'clf' step with attribute classes_
    classes = clf.named_steps["clf"].classes_

    p_tfidf_raw = clf.predict_proba([args.text])[0]
    p_tfidf = expand_proba(p_tfidf_raw, classes)

    if args.model == "tfidf_lr":
        pred = int(np.argmax(p_tfidf))
        print(pred, LABELS.get(pred,"UNKNOWN"))
        return

    rule_lbl = rule_id(args.text)
    p_rules = onehot(rule_lbl)
    p_final = 0.6 * p_tfidf + 0.4 * p_rules

    pred_final = int(np.argmax(p_final))
    print(pred_final, LABELS.get(pred_final,"UNKNOWN"))


if __name__ == "__main__":
    main()
