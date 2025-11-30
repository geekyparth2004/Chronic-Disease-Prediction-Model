import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
csv_path = "Disease_symptom_and_patient_profile_dataset.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    exit(1)

# Preprocessing
if "Disease" in df.columns:
    df = df.drop(columns=["Disease"])

categorical_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                    'Gender', 'Blood Pressure', 'Cholesterol Level']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

# Encode target
label_encoder = LabelEncoder()
if 'Outcome Variable' in df.columns:
    df['Outcome Variable'] = label_encoder.fit_transform(df['Outcome Variable'])
else:
    print("Error: 'Outcome Variable' column not found.")
    exit(1)

# Split and Scale
X = df.drop(columns=['Outcome Variable'])
y = df['Outcome Variable']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest (Baseline)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("-" * 30)
print("Random Forest (Baseline)")
print(f"Accuracy: {accuracy_rf}")
print(classification_report(y_test, y_pred_rf))

# Train Optimized Logistic Regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

print("-" * 30)
print("Optimizing Logistic Regression...")

# Create pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42, max_iter=1000))
])

# Define hyperparameters
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__solver': ['liblinear'], # liblinear supports both l1 and l2
    'logreg__class_weight': [None, 'balanced']
}

# Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_opt = best_model.predict(X_test)
accuracy_opt = accuracy_score(y_test, y_pred_opt)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Optimized Logistic Regression Accuracy: {accuracy_opt}")
print(classification_report(y_test, y_pred_opt))

# Advanced Optimization: RFE and Bagging
from sklearn.feature_selection import RFE
from sklearn.ensemble import BaggingClassifier

print("-" * 30)
print("Advanced Optimization (RFE & Bagging)...")

# 1. RFE with Logistic Regression
# Selecting top 5 features (arbitrary starting point, can be tuned)
rfe_selector = RFE(estimator=LogisticRegression(random_state=42, max_iter=1000), n_features_to_select=5, step=1)
rfe_selector.fit(X_train, y_train)
X_train_rfe = rfe_selector.transform(X_train)
X_test_rfe = rfe_selector.transform(X_test)

lr_rfe = LogisticRegression(random_state=42, max_iter=1000)
lr_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = lr_rfe.predict(X_test_rfe)
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)

print(f"RFE (5 features) + Logistic Regression Accuracy: {accuracy_rfe}")

# 2. Bagging with Logistic Regression
bagging_model = BaggingClassifier(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    n_estimators=50,
    random_state=42
)
bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

print(f"Bagging + Logistic Regression Accuracy: {accuracy_bagging}")

# 3. Bagging with Polynomial Features + LR (Best Combo?)
# We use the pipeline structure from the optimized step
poly_logreg_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42, max_iter=1000, C=0.01, penalty='l2', solver='liblinear')) # Using best params found
])

bagging_poly = BaggingClassifier(
    estimator=poly_logreg_pipeline,
    n_estimators=50,
    random_state=42
)
bagging_poly.fit(X_train, y_train)
y_pred_bagging_poly = bagging_poly.predict(X_test)
accuracy_bagging_poly = accuracy_score(y_test, y_pred_bagging_poly)

print(f"Bagging + Polynomial Features + LR Accuracy: {accuracy_bagging_poly}")
print("-" * 30)

# 4. Optimized Random Forest
print("Optimizing Random Forest...")

# Define hyperparameters for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

# Grid Search
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, n_jobs=-1, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)

best_rf_model = rf_grid_search.best_estimator_
y_pred_rf_opt = best_rf_model.predict(X_test)
accuracy_rf_opt = accuracy_score(y_test, y_pred_rf_opt)

print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")
print(f"Optimized Random Forest Accuracy: {accuracy_rf_opt}")
print(classification_report(y_test, y_pred_rf_opt))
print("-" * 30)
