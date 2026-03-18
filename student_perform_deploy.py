import streamlit as st
import joblib
import pandas as pd

# 1. LOAD THE TRAINED REGRESSION MODEL
# This model has learned the patterns between habits and scores
try:
    model = joblib.load("best_model.pkl")
except:
    st.error("Model file 'best_model.pkl' not found. Please ensure it is in the same folder.")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Score Predictor", layout="wide")
st.title("📈 Student Performance Index Predictor")
st.markdown("""
**Data Analytics Internship Project:** This tool uses a Regression Model to analyze how study habits and history impact a student's final score.
""")

st.divider()

# 2. DATA INPUT SECTION (Prediction Function)
st.subheader("📝 Enter Student Details")
col1, col2, col3 = st.columns(3)

with col1:
    prev_scores = st.number_input("Previous Scores (%)", 0, 100, 75, help="Past academic performance")
    hours_studied = st.number_input("Hours Studied", 0, 24, 8, help="Daily average study hours")
    value_id = st.number_input("Record ID (Value)", value=101)

with col2:
    performance_idx_input = st.number_input("Current Performance Index", 0, 100, 70)
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)
    sample_papers = st.number_input("Sample Papers Practiced", 0, 50, 5)

with col3:
    gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    extra_act = st.selectbox("Extracurricular Activities", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# 3. THE PREDICTION FUNCTION
if st.button("Predict Performance Index", use_container_width=True):
    
    # Mapping inputs to the EXACT feature names the model was trained on
    data_dict = {
        "value": [value_id],
        "Hours Studied": [hours_studied],
        "Previous Scores": [prev_scores],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers],
        "Performance Index": [performance_idx_input], # Included as per your model's 8-feature requirement
        "Gender_Male": [gender],
        "Extracurricular Activities_Yes": [extra_act]
    }

    input_df = pd.DataFrame(data_dict)

    try:
        # Reorder columns to match the training data exactly
        input_ready = input_df[model.feature_names_in_]
        
        # MODEL PREDICTION (Regression)
        prediction = model.predict(input_ready)
        final_score = round(float(prediction[0]), 2)

        # 4. RESULTS & EVALUATION DISPLAY
        st.subheader("🎯 Prediction Result")
        
        # Show the metric
        st.metric(label="Predicted Performance Index", value=f"{final_score}%")
        
        # Visual Progress Bar
        st.progress(min(max(final_score / 100, 0.0), 1.0))

        # Actionable Insight for Teachers/Students
        if final_score >= 80:
            st.success("High Performance: The student is on track for excellence.")
        elif final_score >= 50:
            st.warning("Average Performance: Targeted study sessions could improve the score.")
        else:
            st.error("At-Risk Student: Immediate academic intervention recommended.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# 5. INTERNSHIP FOOTER
st.divider()
st.caption("Data Analytics Internship | Model: Linear Regression | Developer: Shivam Umesh Jaiswal")
