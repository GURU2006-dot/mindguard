from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load dataset and train model
data = pd.read_csv("students.csv")

X = data[["Attendance_%", "Avg_Marks", "Assignment_%", "Stress_Score"]]
y = data["Risk_Level"]

model = DecisionTreeClassifier()
model.fit(X, y)

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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

