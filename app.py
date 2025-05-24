import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load('lung_cancer_model.pkl')
le = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Lung Cancer Survival Prediction")

# User input
st.header("Enter Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
country = st.selectbox("Country", ["USA", "India", "UK", "Germany", "Other"])
cancer_stage = st.selectbox("Cancer Stage", ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
family_history = st.selectbox("Family History of Cancer", ["Yes", "No"])
smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)
cholesterol_level = st.slider("Cholesterol Level", min_value=100, max_value=400, value=200)
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
asthma = st.selectbox("Asthma", ["Yes", "No"])
cirrhosis = st.selectbox("Cirrhosis", ["Yes", "No"])
other_cancer = st.selectbox("Other Cancer", ["Yes", "No"])
treatment_type = st.selectbox("Treatment Type", ['Chemotherapy', 'Radiation', 'Surgery', 'Immunotherapy', 'Other'])

if st.button("Predict Survival"):
    # Create DataFrame for a single row
    input_dict = {
        'age': age,
        'gender': gender,
        'country': country,
        'cancer_stage': cancer_stage,
        'family_history': family_history,
        'smoking_status': smoking_status,
        'bmi': bmi,
        'cholesterol_level': cholesterol_level,
        'hypertension': hypertension,
        'asthma': asthma,
        'cirrhosis': cirrhosis,
        'other_cancer': other_cancer,
        'treatment_type': treatment_type
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical features
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = le.fit_transform(input_df[col])  # Make sure the order of classes matches

    # Scale numerical features
    input_df[['age', 'bmi', 'cholesterol_level']] = scaler.transform(
        input_df[['age', 'bmi', 'cholesterol_level']]
    )

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ The patient is likely to **survive**.")
    else:
        st.error("⚠️ The patient is likely to **not survive**.")
