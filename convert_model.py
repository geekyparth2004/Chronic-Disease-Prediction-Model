import joblib
import xgboost as xgb
import numpy as np

# Load the original model
print("Loading original pickled model...")
original_model = joblib.load("xgboost_chronic_model.pkl")

# Extract the booster
print("Extracting booster...")
booster = original_model.get_booster()

# Save as JSON
json_path = "model.json"
booster.save_model(json_path)
print(f"Model saved to {json_path}")

# Verify 
print("Verifying loading...")
loaded_booster = xgb.Booster()
loaded_booster.load_model(json_path)

# Test prediction (dummy input)
# Note: XGBClassifier.predict calls predict(data), returns classes. 
# Booster.predict(DMatrix) returns probabilities.
# We need to handle this change in app.py.
# Threshold is usually 0.5 for binary classification.

dummy_input = np.random.rand(1, 8)
dmatrix = xgb.DMatrix(dummy_input)

# Original prediction
try:
    orig_pred = original_model.predict(dummy_input)[0]
    orig_prob = original_model.predict_proba(dummy_input)[0][1]
    print(f"Original Prediction: Class {orig_pred}, Prob {orig_prob}")
except Exception as e:
    print(f"Original model prediction failed: {e}")

# New prediction
new_prob = loaded_booster.predict(dmatrix)[0]
new_pred = 1 if new_prob > 0.5 else 0
print(f"New Prediction: Class {new_pred}, Prob {new_prob}")

if abs(orig_prob - new_prob) < 1e-5:
    print("Verification Successful: Predictions match.")
else:
    print("Verification Failed: Predictions differ.")
