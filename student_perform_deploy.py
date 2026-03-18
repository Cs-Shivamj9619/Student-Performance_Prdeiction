import streamlit as st
import joblib
import pandas as pd

# 1. Load the model
model = joblib.load("best_model.pkl")

st.title("Student Performance Prediction")

# 2. User Inputs (Matching the common 5-feature dataset)
# Note: I removed Performance_Index from here because that is usually what we predict.
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
extracurricular = st.selectbox("Extracurricular Activities (0=No, 1=Yes)", [0, 1])
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
papers_practiced = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=50, value=5)

# 3. Create DataFrame for prediction
# IMPORTANT: The order of features MUST match how your model was trained.
# We use model.feature_names_in_ to ensure the columns are named correctly.
input_dict = {
    "Hours Studied": [hours_studied],
    "Previous Scores": [previous_scores],
    "Extracurricular Activities": [extracurricular],
    "Sleep Hours": [sleep_hours],
    "Sample Question Papers Practiced": [papers_practiced]
}

input_data = pd.DataFrame(input_dict)

# Ensure columns match the exact names the model expects
try:
    input_data = input_data[model.feature_names_in_]
except:
    # Fallback if names don't match exactly - just use the values
    pass

# 4. Prediction Logic
if st.button("Predict Result"):
    prediction = model.predict(input_data)
    
    # Extract the result (model.predict returns an array)
    result_value = prediction[0]
    
    st.subheader("Final Verdict:")
    
    # If your model returns 1/0 or "Positive"/"Negative"
    if result_value == 1 or str(result_value).lower() == "positive":
        st.success("POSITIVE")
    else:
        st.error("NEGATIVE")
