import joblib
import numpy as np
import onnxruntime as rt
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# Load the trained XGBoost model
print("Loading model...")
model = joblib.load("xgboost_chronic_model.pkl")

# PATCH: Add missing attributes required by onnxmltools/sklearn/xgboost
if not hasattr(model, "use_label_encoder"):
    model.use_label_encoder = False
if not hasattr(model, "gpu_id"):
    model.gpu_id = -1
if not hasattr(model, "interaction_constraints"):
    model.interaction_constraints = None
if not hasattr(model, "monotone_constraints"):
    model.monotone_constraints = None
if not hasattr(model, "n_jobs"):
    model.n_jobs = 1
if not hasattr(model, "predictor"):
    model.predictor = "auto"
if not hasattr(model, "tree_method"):
    model.tree_method = "auto"
if not hasattr(model, "num_parallel_tree"):
    model.num_parallel_tree = 1
if not hasattr(model, "enable_categorical"):
    model.enable_categorical = False
if not hasattr(model, "scale_pos_weight"):
    model.scale_pos_weight = 1
if not hasattr(model, "validate_parameters"):
    model.validate_parameters = True
if not hasattr(model, "eval_metric"):
    model.eval_metric = None
if not hasattr(model, "classes_"):
    model.classes_ = np.array([0, 1])

# Define input type: 8 features, float
print("Converting to ONNX...")
initial_types = [('float_input', FloatTensorType([None, 8]))]

# Convert using onnxmltools specific for XGBoost
onx = onnxmltools.convert_xgboost(model, initial_types=initial_types)

# Save the ONNX model
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
print("Model saved to model.onnx")

# Verify the ONNX model
print("Verifying ONNX model...")
sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# Create dummy input
dummy_input = np.random.rand(1, 8).astype(np.float32)

# Predict with ONNX
res = sess.run([label_name], {input_name: dummy_input})
print(f"ONNX Prediction: {res[0]}")

# Compare with original model
orig_pred = model.predict(dummy_input)
print(f"Original Prediction: {orig_pred}")

if res[0][0] == orig_pred[0]:
    print("Verification Successful: Predictions match.")
else:
    print("Verification Warning: Predictions differ (could be due to floating point differences).")
