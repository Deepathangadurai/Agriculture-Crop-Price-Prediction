 # Safety check — convert any leftover numeric-like strings to string type
   # df = df.map(lambda x: str(x) if isinstance(x, (int, float)) else x)
import pandas as pd
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_yield_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

def predict_cost(input_data: dict):
    df = pd.DataFrame([input_data])

    # Define categorical and numeric columns (match training features)
    categorical_cols = ['Crop', 'State', 'Season Type', 'Recommended Zone']
    numeric_cols = ['Year', 'Area (ha)', 'Production (tons)', 'Yield (q/ha)']

    # Numeric columns: convert and fill missing values
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Categorical columns: convert to string and fill missing with 'Unknown'
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('Unknown').str.strip()

    # Predict
    predicted_value = model.predict(df)[0]
    return round(predicted_value, 2)

if __name__ == "__main__":
    sample_input = {
        'Crop': 'Rice',
        'State': 'Tamil Nadu',
        'Year': 2025,
        'Area (ha)': 1000,
        'Season Type': 'Kharif',
        'Production (tons)': 500,
        'Yield (q/ha)': 50,
        'Recommended Zone': 'Zone-3'
    }

    result = predict_cost(sample_input)
    print(f"🌾 Predicted Cost: ₹{result}/ha")
