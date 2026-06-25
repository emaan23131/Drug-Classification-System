import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoders
model = joblib.load("drug_model.pkl")

le_sex = joblib.load("le_sex.pkl")
le_bp = joblib.load("le_bp.pkl")
le_cholesterol = joblib.load("le_cholesterol.pkl")
le_drug = joblib.load("le_drug.pkl")

# Page title
st.set_page_config(page_title="Drug Classifier", page_icon="💊")

st.title("💊 Drug Classification System")
st.write("Enter patient information to predict the recommended drug.")

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

    sex_encoded = le_sex.transform([sex])[0]
    bp_encoded = le_bp.transform([bp])[0]
    chol_encoded = le_cholesterol.transform([cholesterol])[0]

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_encoded],
        "BP": [bp_encoded],
        "Cholesterol": [chol_encoded],
        "Na_to_K": [na_to_k]
    })

    prediction = model.predict(input_data)

    predicted_drug = le_drug.inverse_transform(prediction)[0]

    st.success(f"Recommended Drug: {predicted_drug}")