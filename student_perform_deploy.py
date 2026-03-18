import streamlit as st
import joblib
import numpy as np

# 1. Load the model
model = joblib.load("best_model.pkl")

st.title("Student Performance Prediction")

# 2. Input Fields (The 6 features your model likely expects)
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=50, value=5)
# Often datasets include these two binary categories:
tuition = st.selectbox("Tuition (0=No, 1=Yes)", [0, 1])
extracurricular = st.selectbox("Extracurricular Activities (0=No, 1=Yes)", [0, 1])

# 3. Predict Button
if st.button("Predict Result"):
    # Putting exactly 6 features in the list
    features = [
        hours_studied,
        previous_scores,
        sleep_hours,
        sample_papers,
        tuition,
        extracurricular
    ]
    
    try:
        # Convert to 2D array and predict
        prediction = model.predict([features])
        result = prediction[0]

        st.divider()
        st.subheader("Final Result:")

        # Handle string or numeric output
        if str(result).lower() in ['1', 'positive', 'good']:
            st.success("The predicted performance is GOOD")
        else:
            st.error("The predicted performance is BAD")
            
    except ValueError as e:
        st.error(f"Feature count mismatch! Your model might need a different number of inputs.")
        # This will tell us EXACTLY how many features your model wants
        st.info("Debugging Info: Check your model training code to see which columns you used.")
