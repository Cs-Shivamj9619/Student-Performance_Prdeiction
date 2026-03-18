import streamlit as st
import joblib
import pandas as pd

# 1. Load the model
model = joblib.load("best_model.pkl")

st.title("Student Performance Prediction")

# 2. Input Fields (Matched to your model's required list)
# Note: 'value' is likely a placeholder or ID column from your training set.
value_input = st.number_input("Value (ID or Index)", value=0)
hours_studied = st.number_input("Hours Studied", 0, 24, 8)
previous_scores = st.number_input("Previous Scores", 0, 100, 80)
sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)
sample_papers = st.number_input("Sample Question Papers Practiced", 0, 50, 5)
performance_idx = st.number_input("Performance Index (Current)", 0, 100, 70)
gender = st.selectbox("Gender (0=Female, 1=Male)", [1, 0])
extra_activities = st.selectbox("Extracurricular Activities (0=No, 1=Yes)", [1, 0])

# 3. Predict Button
if st.button("Predict Result"):
    # We create the dictionary with the EXACT names shown in your screenshot
    data_dict = {
        "value": [value_input],
        "Hours Studied": [hours_studied],
        "Previous Scores": [previous_scores],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers],
        "Performance Index": [performance_idx],
        "Gender_Male": [gender],
        "Extracurricular Activities_Yes": [extra_activities]
    }

    input_df = pd.DataFrame(data_dict)

    try:
        # This ensures the order is exactly what the model expects
        input_data_reordered = input_df[model.feature_names_in_]
        
        prediction = model.predict(input_data_reordered)
        result = prediction[0]

        st.divider()
        st.subheader("Prediction Result:")

        # Logic to determine Good vs Bad
        # If your model returns 1/0 or a high/low score
        if result == 1 or (isinstance(result, (int, float)) and result > 50):
            st.success(f"Outcome: GOOD (Result: {result})")
        else:
            st.error(f"Outcome: BAD (Result: {result})")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
