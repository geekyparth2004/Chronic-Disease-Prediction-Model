import pandas as pd
import os

# Define file paths
input_csv_path = "Disease_symptom_and_patient_profile_dataset.csv"
output_csv_path = "cleaned_dataset.csv"

# Check if input file exists
if not os.path.exists(input_csv_path):
    print(f"Error: Input file '{input_csv_path}' not found.")
    exit(1)

# Load dataset
print(f"Loading dataset from {input_csv_path}...")
try:
    df = pd.read_csv(input_csv_path)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Display initial stats
print("-" * 30)
print("Initial Dataset Statistics:")
print("Null values per column:")
print(df.isnull().sum())

# Remove null values
print("-" * 30)
print("Removing null values...")
df_cleaned = df.dropna()

# Display stats after cleaning
print("-" * 30)
print("Cleaned Dataset Statistics:")
print("Null values per column:")
print(df_cleaned.isnull().sum())

# Calculate rows removed
rows_removed = df.shape[0] - df_cleaned.shape[0]
print("-" * 30)
print(f"Total rows removed: {rows_removed}")

# Save cleaned dataset
try:
    df_cleaned.to_csv(output_csv_path, index=False)
    print(f"Cleaned dataset saved to '{output_csv_path}'")
except Exception as e:
    print(f"Error saving cleaned CSV: {e}")
    exit(1)
print("-" * 30)
