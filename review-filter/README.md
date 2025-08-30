Overview: problem, labels, approach diagram.

Data: sources, cleaning steps, splits.

Models: rules, TF-IDF+LR, DistilBERT, ensemble.

Training: exact commands.

Evaluation: table with macro-F1 and per-class metrics; link to figures.

Reproduce: 
pip install -r requirements.txt python 
python src/00_inspect.py
python src/01_clean.py
python src/02_rules.py
python src/03_train_tfidf_lr.py
python src/04_train_distilbert.py
python src/05_pseudolabel_llm.py
python src/06_ensemble.py
python src/06c_ensemble_triple.py
python src/07_eval.py
python src/08_demo_infer.py --text "This is a test review"
python src/08c_demo_infer_triple.py --text "This is a test review"
python src/fix_headers.py (only needed if your raw CSV headers are messy; run it separately if required)

Team roles (if any).

Limitations & Next steps.
