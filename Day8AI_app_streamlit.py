import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the input feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Streamlit UI
st.set_page_config(page_title="Heart Risk Predictor", layout="centered")
st.title('💓 Heart Disease Risk Predictor')
st.markdown('Fill in the patient details below:')

# Collect input
inputs = []
for feature in feature_names:
    value = st.number_input(label=feature, step=1.0, format="%.2f")
    inputs.append(value)

# Predict button
if st.button('Predict'):
    input_df = pd.DataFrame([inputs], columns=feature_names)

    # Scale the inputs
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ At Risk: Please consult a doctor.")
    else:
        st.success("✅ No Heart Disease Detected.")
