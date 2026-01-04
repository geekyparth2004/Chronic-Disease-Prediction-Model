from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
# Load model using ONNX (Lightweight, fits in Vercel)
import onnxruntime as rt
sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# Scaler Parameters (Extracted manually to avoid installing scikit-learn on Vercel)
SCALER_MEAN = [0.50143266, 0.47851003, 0.69340974, 0.252149, 46.32378223, 0.49570201, 0.99140401, 0.9512894]
SCALER_SCALE = [0.49999795, 0.49953797, 0.46107773, 0.43424634, 13.06632986, 0.49998153, 0.97383273, 0.9487919]

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

        # Manual Scaling
        input_array = np.array(encoded_input)
        scaled_input = (input_array - SCALER_MEAN) / SCALER_SCALE
        
        # Reshape for model (1, -1)
        scaled_input = scaled_input.reshape(1, -1)
        
        # Reshape for model (1, -1) and ensure float32
        scaled_input = scaled_input.reshape(1, -1).astype(np.float32)
        
        # Predict using ONNX
        prediction = sess.run([label_name], {input_name: scaled_input})[0][0]
        
        prediction_result = f"{name}, based on your inputs, the prediction is: {'Yes (Chronic Disease)' if prediction == 1 else 'No (Healthy)'}"

    return render_template("form.html", prediction_result=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
