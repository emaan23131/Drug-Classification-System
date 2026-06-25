import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("knn_model.pkl")
encoders = joblib.load("encoders.pkl")
drug_encoder = joblib.load("drug_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Page settings
st.set_page_config(
    page_title="Drug Classification System",
    page_icon="💊"
)

st.title("💊 Drug Classification System")
st.write("Enter patient details to predict the recommended drug.")

# Inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["F", "M"]
)

bp = st.selectbox(
    "Blood Pressure",
    ["LOW", "NORMAL", "HIGH"]
)

cholesterol = st.selectbox(
    "Cholesterol",
    ["NORMAL", "HIGH"]
)

na_to_k = st.number_input(
    "Na_to_K Ratio",
    min_value=0.0,
    value=15.0
)

if st.button("Predict Drug"):

    sex_encoded = encoders["Sex"].transform([sex])[0]
    bp_encoded = encoders["BP"].transform([bp])[0]
    chol_encoded = encoders["Cholesterol"].transform([cholesterol])[0]

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_encoded],
        "BP": [bp_encoded],
        "Cholesterol": [chol_encoded],
        "Na_to_K": [na_to_k]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    predicted_drug = drug_encoder.inverse_transform(prediction)[0]

    st.success(f"💊 Recommended Drug: {predicted_drug}")
