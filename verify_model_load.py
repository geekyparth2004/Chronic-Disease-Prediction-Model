import xgboost as xgb
import numpy as np

try:
    print(f"XGBoost Version: {xgb.__version__}")
    model = xgb.Booster()
    model.load_model("model.json")
    print("Model loaded successfully.")
    
    # Test prediction
    dummy_input = np.random.rand(1, 8)
    dmatrix = xgb.DMatrix(dummy_input)
    prediction = model.predict(dmatrix)
    print(f"Prediction: {prediction}")
    
except Exception as e:
    print(f"Error: {e}")
