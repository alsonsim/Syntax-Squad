Pre-requisites: Python 3.8+, Node.js 16+ with npm, git (to clone repo)

Clone the repo: git clone https://github.com/YOUR-USERNAME/Syntax-Squad.git
                          cd Syntax-Squad/review-filter

Launch the backend (be in review-filter/ folder): uvicorn src.app:app --reload --app-dir src
                                                  Visit http://127.0.0.1:8000/docs to confirm the API is running.

Start the React app: npm start
