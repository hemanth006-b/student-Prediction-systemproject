"""Flask app for Vercel. Vercel auto-detects `app` in app.py."""
from pathlib import Path
import csv
import numpy as np
import joblib
from flask import Flask, Response, jsonify, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_performance_model.pkl"
DATA_PATH = ROOT / "data" / "student_performance.csv"
FEATURES = ["study_hours", "attendance", "previous_marks", "assignment_score"]

app = Flask(__name__)

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Performance Predictor</title>
<style>
  :root { --bg:#f6f7fb; --card:#fff; --text:#1f2430; --muted:#6b7280; --accent:#4f46e5; --border:#e5e7eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12141b; --card:#1b1e28; --text:#e8eaf0; --muted:#9aa1b2; --accent:#818cf8; --border:#2c3040; }
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  main { max-width:720px; margin:0 auto; padding:24px 16px 48px; }
  h1 { margin:0 0 4px; font-size:1.7rem; }
  p.sub { color:var(--muted); margin:0 0 20px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:20px; margin-bottom:16px; }
  label { display:block; font-weight:600; margin:14px 0 6px; }
  label span { float:right; color:var(--accent); }
  input[type=text] { width:100%; padding:10px; border-radius:8px; border:1px solid var(--border); background:transparent; color:inherit; }
  input[type=range] { width:100%; accent-color:var(--accent); }
  button { margin-top:20px; width:100%; padding:12px; border:0; border-radius:10px; background:var(--accent); color:#fff; font-size:1rem; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.6; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .metric { background:var(--bg); border-radius:10px; padding:12px; }
  .metric small { color:var(--muted); display:block; }
  .metric strong { font-size:1.4rem; }
  .bar { height:10px; border-radius:6px; background:var(--border); overflow:hidden; margin:14px 0; }
  .bar div { height:100%; width:0; background:var(--accent); transition:width .4s; }
  .msg { padding:12px; border-radius:10px; }
  .ok { background:rgba(34,197,94,.15); } .warn { background:rgba(234,179,8,.2); } .err { background:rgba(239,68,68,.15); }
  .note { color:var(--muted); font-size:.85rem; }
  @media (max-width:520px) { .grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
  <h1>🎓 Student Performance Prediction</h1>
  <p class="sub">An AI/ML mini project that predicts a student's expected final marks.</p>

  <form id="f" class="card">
    <label for="name">Student name (optional)</label>
    <input type="text" id="name" placeholder="Enter student name">

    <label for="study_hours">Daily study hours <span id="v1">3</span></label>
    <input type="range" id="study_hours" min="0" max="12" step="0.5" value="3">

    <label for="attendance">Attendance (%) <span id="v2">80</span></label>
    <input type="range" id="attendance" min="0" max="100" step="1" value="80">

    <label for="previous_marks">Previous exam marks (%) <span id="v3">65</span></label>
    <input type="range" id="previous_marks" min="0" max="100" step="1" value="65">

    <label for="assignment_score">Assignment score (%) <span id="v4">70</span></label>
    <input type="range" id="assignment_score" min="0" max="100" step="1" value="70">

    <button type="submit" id="btn">Predict Performance</button>
  </form>

  <section id="result" class="card" hidden>
    <h2 id="title" style="margin-top:0"></h2>
    <div class="grid">
      <div class="metric"><small>Predicted final marks</small><strong id="marks"></strong></div>
      <div class="metric"><small>Performance category</small><strong id="cat"></strong></div>
    </div>
    <div class="bar"><div id="bar"></div></div>
    <div id="advice" class="msg"></div>
    <p class="note">This is an estimate from a model trained on sample data, not an official grade.</p>
  </section>

  <section class="card">
    <h3 style="margin-top:0">About this project</h3>
    <p class="note" style="font-size:.95rem">Algorithm: Random Forest Regression. Inputs: study hours, attendance, previous marks and assignment score. Output: predicted final examination marks and a performance category.</p>
  </section>
</main>
<script>
  const ids = ["study_hours","attendance","previous_marks","assignment_score"];
  ids.forEach((id,i) => {
    const el = document.getElementById(id), out = document.getElementById("v"+(i+1));
    el.addEventListener("input", () => out.textContent = el.value);
  });
  document.getElementById("f").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("btn");
    btn.disabled = true; btn.textContent = "Predicting…";
    const body = {}; ids.forEach(id => body[id] = Number(document.getElementById(id).value));
    const adv = document.getElementById("advice");
    try {
      const r = await fetch("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Request failed");
      const name = document.getElementById("name").value.trim() || "Student";
      document.getElementById("title").textContent = "Prediction for " + name;
      document.getElementById("marks").textContent = d.predicted.toFixed(1) + " / 100";
      document.getElementById("cat").textContent = d.category;
      document.getElementById("bar").style.width = d.predicted + "%";
      adv.textContent = d.advice;
      adv.className = "msg " + (d.support ? "warn" : "ok");
      document.getElementById("result").hidden = false;
    } catch (err) {
      document.getElementById("result").hidden = false;
      adv.textContent = err.message; adv.className = "msg err";
    } finally {
      btn.disabled = false; btn.textContent = "Predict Performance";
    }
  });
</script>
</body>
</html>
"""

_model = None


def _train_fallback():
    """Train in memory if the saved model can't be loaded (e.g. version mismatch)."""
    from sklearn.ensemble import RandomForestRegressor
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    y = np.array([float(r["final_marks"]) for r in rows])
    return RandomForestRegressor(n_estimators=150, random_state=42).fit(X, y)


def get_model():
    """Load the model on first use so the home page never depends on it."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            _model = _train_fallback()
    return _model


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
    return Response(PAGE, mimetype="text/html")


@app.get("/health")
def health():
    return jsonify(status="ok")


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
    try:
        pred = float(get_model().predict(np.array([vals]))[0])
    except Exception as e:
        return jsonify(error=f"Model unavailable: {e}"), 500
    pred = max(0.0, min(100.0, pred))
    cat, advice, support = categorize(pred, att)
    return jsonify(predicted=round(pred, 1), category=cat, advice=advice, support=support)


if __name__ == "__main__":
    app.run(debug=True)
