# Student Performance Prediction System (AI & ML)

A beginner-friendly AI/ML mini project that predicts a student's expected final marks
using Random Forest Regression, classifies the result (Excellent / Good / Average /
Needs Improvement) and gives a short study suggestion.

## Project structure
```text
├── app.py                          # Flask web app (used for Vercel)
├── templates/index.html            # Web UI
├── streamlit_app.py                # Streamlit version (local / Streamlit Cloud)
├── train_model.py                  # Trains and saves the model
├── generate_data.py                # Creates the synthetic dataset
├── data/student_performance.csv
├── student_performance_model.pkl   # Trained model (scikit-learn 1.8.0)
├── requirements.txt                # Deployment dependencies (Vercel)
├── requirements-dev.txt            # Local extras (pandas, streamlit)
├── run.bat                         # Windows shortcut for Streamlit
└── .vercelignore / .gitignore
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt

python generate_data.py          # optional: regenerate dataset
python train_model.py            # optional: retrain model

python app.py                    # Flask version  -> http://127.0.0.1:5000
streamlit run streamlit_app.py   # Streamlit version -> http://localhost:8501
```

## Deploy on Vercel
1. Push this repo to GitHub.
2. On vercel.com choose **Add New > Project**, import the repo, keep the defaults
   (Vercel detects Flask from `app.py`) and click **Deploy**.

If you retrain the model, keep the `scikit-learn` version in `requirements.txt`
the same as the one you trained with. If the saved model can't be loaded, the app
retrains itself from `data/student_performance.csv` automatically.

## Deploy the Streamlit version
Vercel cannot host Streamlit. Use [Streamlit Community Cloud](https://share.streamlit.io),
select this repo and set the main file to `streamlit_app.py`.

## Note
The dataset is synthetic and for learning/demo purposes only. Predictions must not
be used alone for high-stakes student decisions.
