Review Category Classifier is a web app that classifies google location reviews using an ensemble model of rule-based logic, TF-IDF, and Hugging Face's DistilBERT model. 

Features

-Web UI for easy review submission and classification

-FastAPI backend (Python) with modular, extensible ML logic

-Combines classic ML and deep learning for robust predictions

-Ready for local and cloud deployment



Prerequisites

-Python 3.8+

-Node.js 16+ (with npm)

-git (to clone repository)



🚀 Quickstart

1. Clone the repository:

   git clone https://github.com/YOUR-USERNAME/Syntax-Squad.git

   cd Syntax-Squad/review-filter

3. Set up and launch FastAPI backend:

   python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

Place trained TF-IDF model at: models/tfidf_lr/model.joblib
Place DistilBERT transformer files in: models/distilbert/

Start the backend server:

uvicorn src.app:app --reload

Visit http://127.0.0.1:8000/docs to check the API is live.

3. Set up and start React frontend


npx create-react-app review-ui

cd review-ui

npm install

Replace review-ui/src/App.js with:

import React, { useState } from 'react';

function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: review }),
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setResult({ error: "There was an error connecting to the server." });
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', fontFamily: 'Arial' }}>
      <h1>Review Category Classifier</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={review}
          onChange={e => setReview(e.target.value)}
          rows={5}
          style={{ width: '100%', marginBottom: 12 }}
          placeholder="Type your review here..."
        />
        <br />
        <button type="submit" disabled={loading || !review}>
          {loading ? 'Classifying...' : 'Classify'}
        </button>
      </form>
      {result && (
        <div style={{
          background: '#fafafa', padding: 20, marginTop: 24,
          border: '1px solid #ddd', borderRadius: 6
        }}>
          {result.error
            ? <span style={{ color: 'red' }}>{result.error}</span>
            : (
              <>
                <strong>Category:</strong> {result.label}<br />
                <strong>Category ID:</strong> {result.label_id}<br />
                {"confidence" in result ? <span><strong>Confidence:</strong> {result.confidence.toFixed(3)}</span> : null}
              </>
            )
          }
        </div>
      )}
    </div>
  );
}

export default App;


Start your frontend:

npm start
Go to http://localhost:3000 in your browser.

Usage
1. Start both FastAPI (step 2) and React (step 3)

2. Enter a review (e.g., The food was great and service prompt!) and click Classify

3. See predicted category, ID, and confidence score

Troubleshooting
"npm error missing script: start" ->	cd review-ui before running npm start
'react-scripts' not recognized ->	Run npm install in review-ui
"error connecting to the server" ->	Ensure FastAPI backend running and URL matches
"ModuleNotFoundError" on backend ->	Match import statement & filename
Only "Unknown or unable to classify" ->	Backend might not be using models/rules ensemble

Deployment

Frontend:
In review-ui/, build for production:
npm run build
Deploy /build with Netlify, Vercel, or GitHub Pages.

Backend:
Host FastAPI with Render, Railway, or a cloud VM.

Update API URL in React (App.js) to match your backend.

Example Input
"The food was delicious and the staff were friendly!"

Credits
Built with FastAPI & React
Developed by [Jian Wei]
