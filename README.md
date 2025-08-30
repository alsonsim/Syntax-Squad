Review Category Classifier is a web app that classifies google location reviews using an ensemble model of rule-based logic, TF-IDF, and Hugging Face's DistilBERT model. 

# Features

-Web UI for easy review submission and classification

-FastAPI backend (Python) with modular, extensible ML logic

-Combines classic ML and deep learning for robust predictions

-Ready for local and cloud deployment



# Prerequisites

-Python 3.8+

-Node.js 16+ (with npm)

-git (to clone repository)



# Quickstart

1. Clone the repository:

   git clone https://github.com/YOUR-USERNAME/Syntax-Squad.git

   cd Syntax-Squad/review-filter

2. Set up and launch FastAPI backend:

   python -m venv venv

   On Windows:
         venv\Scripts\activate

   On Mac/Linux:
         source venv/bin/activate

   pip install --upgrade pip

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

   Start the backend server:

      uvicorn src.app:app --reload

      Visit http://127.0.0.1:8000/docs to check the API is live.

3. Set up and start React frontend

   npx create-react-app review-ui

   cd review-ui

   npm install

   npm start
   
   Go to http://localhost:3000 in your browser.


# In the web app: 

1. Enter a review (e.g., The food was great and service prompt!) and click Classify

2. See predicted category, ID, and confidence score

# Troubleshooting

"npm error missing script: start" ->	cd review-ui before running npm start

'react-scripts' not recognized ->	Run npm install in review-ui

"error connecting to the server" ->	Ensure FastAPI backend running and URL matches

"ModuleNotFoundError" on backend ->	Match import statement & filename

Only "Unknown or unable to classify" ->	Backend might not be using models/rules ensemble



Credits
Built with FastAPI & React
Developed by [Syntax Squad]
