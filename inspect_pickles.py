import joblib
import numpy as np

try:
    print("Inspecting scaler.pkl...")
    scaler = joblib.load("scaler.pkl")
    print(f"Scaler type: {type(scaler)}")
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        print("Scaler Mean:", scaler.mean_)
        print("Scaler Scale:", scaler.scale_)
    else:
        print("Scaler does not have mean_ or scale_ attributes.")

    print("\nInspecting xgboost_chronic_model.pkl...")
    model = joblib.load("xgboost_chronic_model.pkl")
    print(f"Model type: {type(model)}")
    try:
        import xgboost
        print("XGBoost version:", xgboost.__version__)
    except ImportError:
        print("XGBoost not installed.")

except Exception as e:
    print(f"Error inspecting files: {e}")
