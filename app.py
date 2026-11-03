"""Flask app for Vercel. Vercel auto-detects `app` in app.py."""
from pathlib import Path
import csv
import numpy as np
import joblib
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_performance_model.pkl"
DATA_PATH = ROOT / "data" / "student_performance.csv"
FEATURES = ["study_hours", "attendance", "previous_marks", "assignment_score"]

app = Flask(__name__, template_folder=str(ROOT / "templates"))


def _train_fallback():
    """Train in memory if the saved model can't be loaded (e.g. version mismatch)."""
    from sklearn.ensemble import RandomForestRegressor
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    y = np.array([float(r["final_marks"]) for r in rows])
    return RandomForestRegressor(n_estimators=150, random_state=42).fit(X, y)


def _load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return _train_fallback()


model = _load_model()


def categorize(p, attendance):
    if p >= 85:
        cat, advice = "Excellent", "Keep up the consistent study habits."
    elif p >= 70:
        cat, advice = "Good", "Revise regularly and continue practising."
    elif p >= 50:
        cat, advice = "Average", "Increase focused study time and assignment practice."
    else:
        cat, advice = "Needs Improvement", "Create a study timetable and seek academic support."
    support = attendance < 65 or p < 50
    if support:
        advice = "Academic support recommended: review attendance and study habits."
    return cat, advice, support


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        vals = [float(data[k]) for k in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Provide numeric: " + ", ".join(FEATURES)), 400
    study, att, prev, assign = vals
    if not (0 <= study <= 12 and all(0 <= v <= 100 for v in (att, prev, assign))):
        return jsonify(error="Values out of range."), 400
    pred = float(model.predict(np.array([vals]))[0])
    pred = max(0.0, min(100.0, pred))
    cat, advice, support = categorize(pred, att)
    return jsonify(predicted=round(pred, 1), category=cat, advice=advice, support=support)


if __name__ == "__main__":
    app.run(debug=True)
