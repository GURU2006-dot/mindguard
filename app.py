from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os

# -----------------------------
# Setup paths safely
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "students.csv")

# -----------------------------
# Initialize Flask app
# -----------------------------

app = Flask(__name__, template_folder="templates")

# -----------------------------
# Load dataset
# -----------------------------

data = pd.read_csv(csv_path)

# -----------------------------
# Train AI model
# -----------------------------

X = data[["Attendance_%", "Avg_Marks", "Assignment_%", "Stress_Score"]]
y = data["Risk_Level"]

model = DecisionTreeClassifier()
model.fit(X, y)

# -----------------------------
# Home Page (Prediction Form)
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        attendance = float(request.form["attendance"])
        marks = float(request.form["marks"])
        assignment = float(request.form["assignment"])
        stress = float(request.form["stress"])

        new_data = pd.DataFrame([{
            "Attendance_%": attendance,
            "Avg_Marks": marks,
            "Assignment_%": assignment,
            "Stress_Score": stress
        }])

        prediction = model.predict(new_data)[0]

        if prediction == 0:
            result = "LOW RISK"
        elif prediction == 1:
            result = "MEDIUM RISK"
        else:
            result = "HIGH RISK"

    return render_template("index.html", result=result)

# -----------------------------
# Mentor Dashboard Page
# -----------------------------

@app.route("/dashboard")
def dashboard():

    dashboard_data = data.copy()

    predictions = model.predict(
        dashboard_data[["Attendance_%", "Avg_Marks", "Assignment_%", "Stress_Score"]]
    )

    risk_labels = []

    for p in predictions:

        if p == 0:
            risk_labels.append("LOW")

        elif p == 1:
            risk_labels.append("MEDIUM")

        else:
            risk_labels.append("HIGH")

    dashboard_data["Risk"] = risk_labels

    return render_template(
        "dashboard.html",
        students=dashboard_data.to_dict(orient="records")
    )

# -----------------------------
# Run server (Render compatible)
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
