from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

class ReviewClassifier:
    def __init__(self):
        self.load_models()
        self.setup_rules()

    def load_models(self):
        # Load TF-IDF model
        tfidf_path = Path("models/tfidf_lr/model.joblib")
        self.tfidf = joblib.load(tfidf_path)
        self.tfidf_classes = self.tfidf.named_steps["clf"].classes_

        # Load DistilBERT model
        bert_dir = Path("models/distilbert")
        self.tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
        self.bert_model.eval()

