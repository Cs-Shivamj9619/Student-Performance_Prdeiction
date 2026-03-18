import streamlit as st
import joblib
import pandas as pd

# 1. Load the model
# Make sure the file 'best_model.pkl' is in the same folder as this script
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Student Performance Pro", layout="centered")
st.title("🎓 Student Performance Prediction")
st.write("Enter the details below to see the predicted outcome.")

# 2. Input Fields (Matched exactly to your model's 8 required features)
col1, col2 = st.columns(2)

with col1:
    value_idx = st.number_input("Value (ID/Index)", value=100)
    hours_studied = st.number_input("Hours Studied", 0, 24, 15)
    prev_scores = st.number_input("Previous Scores", 0, 100, 90)
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 8)

with col2:
    sample_papers = st.number_input("Papers Practiced", 0, 50, 10)
    performance_idx = st.number_input("Performance Index", 0, 100, 85)
    gender = st.selectbox("Gender (0=Female, 1=Male)", [1, 0])
    extra_act = st.selectbox("Extracurricular (0=No, 1=Yes)", [1, 0])

st.divider()

# 3. Prediction Logic
if st.button("Predict Result", use_container_width=True):
    # Create the dictionary with the EXACT names from your error screen
    data_dict = {
        "value": [value_idx],
        "Hours Studied": [hours_studied],
        "Previous Scores": [prev_scores],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers],
        "Performance Index": [performance_idx],
        "Gender_Male": [gender],
        "Extracurricular Activities_Yes": [extra_act]
    }

    input_df = pd.DataFrame(data_dict)

    try:
        # Reorder columns to match the model's training order automatically
        input_data_ready = input_df[model.feature_names_in_]
        
        # Get the prediction
        prediction = model.predict(input_data_ready)
        result = prediction[0]

        st.subheader("Prediction Result:")
        
        # IMPORTANT: If '1' is Good, use this. If '0' is Good, swap the labels.
        if result == 1:
            st.success(f"✅ Outcome: GOOD (Model Output: {result})")
        else:
            st.error(f"❌ Outcome: BAD (Model Output: {result})")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.caption("Developed by Shivam | Computer Science Project")
