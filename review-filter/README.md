Overview: problem, labels, approach diagram.

Data: sources, cleaning steps, splits.

Models: rules, TF-IDF+LR, DistilBERT, ensemble.

Training: exact commands.

Evaluation: table with macro-F1 and per-class metrics; link to figures.

Reproduce:
    pip install -r requirements.txt
    python src/01_clean.py
    python src/03_train_tfidf_lr.py
    python src/04_train_distilbert.py
    python src/06_ensemble.py
    python src/07_eval.py
    python src/08_demo_infer.py --model ensemble --text "sample"

Team roles:

Limitations & Next steps: