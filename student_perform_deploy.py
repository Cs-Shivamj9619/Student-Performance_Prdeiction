import streamlit as st
import joblib
import numpy as np

# 1. Load the model
model = joblib.load("best_model.pkl")

st.title("Student Performance Prediction")

# 2. Input Fields (Providing 7 features)
hours_studied = st.number_input("Hours Studied", 0, 24, 5)
previous_scores = st.number_input("Previous Scores", 0, 100, 70)
sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)
sample_papers = st.number_input("Sample Question Papers Practiced", 0, 50, 5)
tuition = st.selectbox("Tuition (0=No, 1=Yes)", [0, 1])
extracurricular = st.selectbox("Extracurricular Activities (0=No, 1=Yes)", [0, 1])

# This is likely your 7th feature (Adjust label if your dataset used something else)
parental_level = st.selectbox("Parental Education Level (0=Low, 1=High)", [0, 1])

# 3. Predict Button
if st.button("Predict Result"):
    # This list MUST have exactly 7 items
    features = [
        hours_studied,
        previous_scores,
        sleep_hours,
        sample_papers,
        tuition,
        extracurricular,
        parental_level  # The 7th feature
    ]
    
    try:
        prediction = model.predict([features])
        result = prediction[0]

        st.divider()
        st.subheader("Final Result:")

        # Using a simple check for common outputs
        if str(result).lower() in ['1', 'positive', 'good', '1.0']:
            st.success("Performance: GOOD")
        else:
            st.error("Performance: BAD")
            
    except Exception as e:
        st.error(f"Error: {e}")
