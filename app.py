from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("xgboost_chronic_model.pkl")
scaler = joblib.load("scaler.pkl")

# Encoders
gender_map = {"Male": 1, "Female": 0}
ethnicity_map = {"Asian": 0, "Black": 1, "White": 2, "Hispanic": 3, "Other": 4}
yes_no_map = {"Yes": 1, "No": 0}
bp_map = {"Low": 0, "Normal": 1, "High": 2}
cholesterol_map = {"Low": 0, "Normal": 1, "High": 2}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    if request.method == "POST":
        # Collect form data
        name = request.form.get("name")
        age = int(request.form.get("age"))
        gender = request.form.get("gender")
        ethnicity = request.form.get("ethnicity")
        fever = request.form.get("fever")
        cough = request.form.get("cough")
        fatigue = request.form.get("fatigue")
        bp = request.form.get("bp")
        cholesterol = request.form.get("cholesterol")

        # Prepare input
        encoded_input = [
            age,
            gender_map.get(gender, 0),
            ethnicity_map.get(ethnicity, 4),
            yes_no_map.get(fever, 0),
            yes_no_map.get(cough, 0),
            yes_no_map.get(fatigue, 0),
            bp_map.get(bp, 1),
            cholesterol_map.get(cholesterol, 1)
        ]

        scaled_input = scaler.transform([encoded_input])
        prediction = model.predict(scaled_input)[0]
        prediction_result = f"{name}, based on your inputs, the prediction is: {'Yes (Chronic Disease)' if prediction == 1 else 'No (Healthy)'}"

    return render_template("form.html", prediction_result=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
