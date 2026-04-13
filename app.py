import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# 2. Load the trained model
@st.cache_resource
def load_model():
    with open('../models/diabetes_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 3. App Header
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's data below to predict if they have diabetes.")

# 4. Input Fields (Based on our Dataset features)
st.sidebar.header("Patient Information")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=250, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# 5. Prediction Logic
if st.button("Predict Result"):
    # Arrange inputs into a numpy array for the model
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)
    
    st.subheader("Results:")
    if prediction[0] == 1:
        st.error("🚨 The model predicts: **Diabetes Positive**")
    else:
        st.success("✅ The model predicts: **Diabetes Negative (Healthy)**")

# 6. Footer
st.markdown("---")
st.write("Developed for Machine Learning Project 2026")