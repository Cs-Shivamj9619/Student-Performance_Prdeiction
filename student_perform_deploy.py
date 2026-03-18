import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the model
try:
    model = joblib.load("best_model.pkl")
    # Identify how many features the model was trained with
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
    else:
        # Default fallback for most student datasets
        expected_features = 5 
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Student Performance Prediction")
st.write(f"Note: This model is configured to use {expected_features} input features.")

# 2. Define all possible inputs
# We define 6 common ones, but we will only send what the model asks for.
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=50, value=5)
tuition = st.selectbox("Tuition (0=No, 1=Yes)", [0, 1])
extracurricular = st.selectbox("Extracurricular Activities (0=No, 1=Yes)", [0, 1])

# 3. Predict Button
if st.button("Predict Result"):
    # Create a master list of all possible features
    all_features = [
        hours_studied,
        previous_scores,
        sleep_hours,
        sample_papers,
        tuition,
        extracurricular
    ]
    
    # SLICE the list to exactly the number the model wants
    # If the model wants 5, it takes the first 5. If 6, it takes all 6.
    final_features = all_features[:expected_features]
    
    try:
        # Perform prediction
        prediction = model.predict([final_features])
        result = prediction[0]

        st.divider()
        st.subheader("Result:")

        # Handle numeric (0/1) or string ("Positive"/"Negative") outputs
        # This logic works for both Classifiers and Regressors
        if str(result).lower() in ['1', 'positive', 'good', '1.0'] or (isinstance(result, (int, float)) and result > 0.5):
            st.success(f"SUCCESS: The predicted performance is POSITIVE (Value: {result})")
        else:
            st.error(f"NOTICE: The predicted performance is NEGATIVE (Value: {result})")
            
    except ValueError as e:
        st.error("Feature Mismatch Error!")
        st.info(f"The model expects {expected_features} features, but the input list logic failed.")
        st.write(f"Technical details: {e}")

# Footer for your college project
st.sidebar.info("Developed by Shivam Jaiswal")
