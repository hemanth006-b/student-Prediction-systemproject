"""
Train a student final-marks prediction model.
Run: python train_model.py
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_PATH = ROOT / "student_performance_model.pkl"

def main():
    data = pd.read_csv(DATA_PATH)
    features = ["study_hours", "attendance", "previous_marks", "assignment_score"]
    X = data[features]
    y = data["final_marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Model training completed.")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R2 Score: {r2_score(y_test, predictions):.3f}")

    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
