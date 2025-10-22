import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_yield_model.pkl")
model = joblib.load(MODEL_PATH)

# -----------------------------
# Page Layout
# -----------------------------
st.set_page_config(
    page_title="🌾 Crop Cost Predictor",
    page_icon="🌱",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: green;'>🌾 Crop Cost Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter crop details to predict <b>Cost per hectare (₹/ha)</b></p>", unsafe_allow_html=True)
st.write("---")

# -----------------------------
# Inputs in Columns
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    crop = st.selectbox("🌾 Crop", ["Rice", "Wheat", "Maize", "Cotton"])
    state = st.selectbox("🏞️ State", ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh"])
    year = st.number_input("📅 Year", min_value=2000, max_value=2050, value=2025)

with col2:
    area = st.number_input("📏 Area (ha)", min_value=1, value=1000)
    season = st.selectbox("🌦️ Season Type", ["Kharif", "Rabi", "Zaid"])
    production = st.number_input("⚖️ Production (tons)", min_value=1, value=500)

with col3:
    yield_q = st.number_input("🌱 Yield (q/ha)", min_value=1, value=50)
    zone = st.selectbox("📍 Recommended Zone", ["Zone-1", "Zone-2", "Zone-3", "Zone-4"])

st.write("---")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("💰 Predict Cost"):
    input_data = {
        'Crop': crop,
        'State': state,
        'Year': year,
        'Area (ha)': area,
        'Season Type': season,
        'Production (tons)': production,
        'Yield (q/ha)': yield_q,
        'Recommended Zone': zone
    }

    # Data Preprocessing
    df = pd.DataFrame([input_data])
    numeric_cols = ['Year', 'Area (ha)', 'Production (tons)', 'Yield (q/ha)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in df.columns:
        if col not in numeric_cols:
            df[col] = df[col].astype(str).fillna("Unknown")

    # Prediction
    predicted_cost = model.predict(df)[0]

    # -----------------------------
    # Display Result in Card Style
    # -----------------------------
    st.markdown(f"""
        <div style='
            background-color: #d4edda;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #155724;
        '>
            🌾 Predicted Cost per hectare: ₹{round(predicted_cost,2)}
        </div>
    """, unsafe_allow_html=True)

    st.info("💡 Note: This is an ML-based prediction. Actual costs may vary based on market and other factors.")
