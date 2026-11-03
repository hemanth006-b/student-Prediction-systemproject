from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_performance_model.pkl"
DATA_PATH = ROOT / "data" / "student_performance.csv"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        from train_model import main
        main()
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🎓 Student Performance Prediction")
st.write("An AI/ML mini project that predicts a student's expected final marks.")
st.info("Enter the student's academic details and click **Predict Performance**.")

with st.form("prediction_form"):
    student_name = st.text_input("Student name (optional)", placeholder="Enter student name")
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Daily study hours", 0.0, 12.0, 3.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
    with col2:
        previous_marks = st.slider("Previous exam marks (%)", 0, 100, 65)
        assignment_score = st.slider("Assignment score (%)", 0, 100, 70)
    submitted = st.form_submit_button("Predict Performance", type="primary")

if submitted:
    sample = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignment_score": assignment_score
    }])
    predicted = float(model.predict(sample)[0])
    predicted = max(0.0, min(100.0, predicted))

    if predicted >= 85:
        category, advice = "Excellent", "Keep up the consistent study habits."
    elif predicted >= 70:
        category, advice = "Good", "Revise regularly and continue practising."
    elif predicted >= 50:
        category, advice = "Average", "Increase focused study time and assignment practice."
    else:
        category, advice = "Needs Improvement", "Create a study timetable and seek academic support."

    st.subheader(f"Prediction for {student_name.strip() or 'Student'}")
    a, b = st.columns(2)
    a.metric("Predicted final marks", f"{predicted:.1f} / 100")
    b.metric("Performance category", category)
    st.progress(int(round(predicted)))
    if attendance < 65 or predicted < 50:
        st.warning("Academic support recommended: review attendance and study habits.")
    else:
        st.success(advice)
    st.caption("This is an estimate from a model trained on sample data, not an official grade.")

st.divider()
st.subheader("About this project")
st.write(
    "Algorithm: Random Forest Regression. "
    "Inputs: study hours, attendance, previous marks and assignment score. "
    "Output: predicted final examination marks and a performance category."
)
with st.expander("View sample dataset"):
    try:
        st.dataframe(pd.read_csv(DATA_PATH).head(10), width="stretch")
    except TypeError:
        st.dataframe(pd.read_csv(DATA_PATH).head(10), use_container_width=True)
