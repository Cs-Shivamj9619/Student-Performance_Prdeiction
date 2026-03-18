import streamlit as st
import joblib
import pandas as pd

# 1. Load the model
# Ensure 'best_model.pkl' is in the same folder as this script
model = joblib.load("best_model.pkl")

# --- UI Setup ---
st.set_page_config(page_title="Student Performance Pro", layout="centered")

st.title("🎓 Student Performance Prediction")
st.markdown("### Computer Science Project - Student Outcome Analysis")
st.write("Fill in the student's details below to predict their performance category.")

st.divider()

# 2. Input Fields (Matched to your model's 8 specific features)
col1, col2 = st.columns(2)

with col1:
    value_idx = st.number_input("Value (Record ID)", value=100)
    hours_studied = st.number_input("Hours Studied", 0, 24, 15)
    prev_scores = st.number_input("Previous Scores (%)", 0, 100, 90)
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 8)

with col2:
    sample_papers = st.number_input("Sample Papers Practiced", 0, 50, 10)
    performance_idx = st.number_input("Performance Index (0-100)", 0, 100, 85)
    gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    extra_act = st.selectbox("Extracurricular Activities", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# 3. Prediction Logic
if st.button("Predict Student Outcome", use_container_width=True):
    # Create the dictionary with the EXACT names your model requires
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
        
        # Get the prediction (0 or 1)
        prediction = model.predict(input_data_ready)
        result = int(prediction[0])

        st.subheader("Final Verdict:")
        
        # --- CORRECTED LOGIC ---
        # Based on your high-performing test case (15 hrs, 90% score), 
        # your model outputs '0' for GOOD students.
        if result == 0:
            st.success(f"✅ Prediction: **GOOD PERFORMANCE** (Model Output: {result})")
            st.balloons()
        else:
            st.error(f"❌ Prediction: **BAD PERFORMANCE** (Model Output: {result})")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.divider()
st.caption("Developed by Shivam Umesh Jaiswal | B.Sc. Computer Science")
