"""Generate a synthetic student dataset. Run: python generate_data.py"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500
study_hours = np.round(rng.uniform(0, 10, n) * 2) / 2
attendance = rng.integers(50, 101, n)
previous_marks = np.clip(rng.normal(65, 15, n), 20, 100).round()
assignment_score = np.clip(previous_marks * 0.5 + rng.normal(35, 12, n), 20, 100).round()

final = (
    0.30 * previous_marks
    + 0.25 * assignment_score
    + 0.20 * attendance
    + 2.5 * study_hours
    + rng.normal(0, 4, n)
)
final_marks = np.clip(final, 0, 100).round(1)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks.astype(int),
    "assignment_score": assignment_score.astype(int),
    "final_marks": final_marks,
})
out = Path(__file__).resolve().parent / "data" / "student_performance.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
