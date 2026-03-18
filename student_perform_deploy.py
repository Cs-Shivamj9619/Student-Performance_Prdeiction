import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the model
model = joblib.load("best_model.pkl")

st.title("Student Performance Prediction")

# 2. Input Fields
# These 5 fields are standard for the Student Performance dataset
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
extracurricular = st.selectbox("Extracurricular Activities (0=No, 1=Yes)", [0, 1])
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
papers_practiced = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=50, value=5)

# 3. Predict Button
if st.button("Predict Result"):
    # We convert inputs to a simple list of numbers
    # The order must be exactly what your model was trained on
    features = [
        hours_studied, 
        previous_scores, 
        extracurricular, 
        sleep_hours, 
        papers_practiced
    ]
    
    # Convert to a 2D array and predict
    # This ignores column names and just looks at the values
    prediction = model.predict([features])
    result = prediction[0]

    st.divider()
    st.subheader("Final Result:")

    # Checking for common result types (1/0 or String)
    if str(result).lower() in ['1', 'positive', 'good']:
        st.success("SUCCESS: Performance is POSITIVE/GOOD")
    else:
        st.error("NOTICE: Performance is NEGATIVE/BAD")
    
    # Optional: Show the raw value if you're curious what the model returned
    # st.write(f"Raw Model Output: {result}")
