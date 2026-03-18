import streamlit as st
import joblib
import pandas as pd

# 1. LOAD THE MODEL
# Keeping this so the 'feature_names_in_' works for background compatibility
model = joblib.load("best_model.pkl")

# --- UI SETUP ---
st.set_page_config(page_title="Student Performance", layout="centered")
st.title("🎓 Student Performance Predictor")
st.write("Academic analysis for B.Sc. Computer Science Students")

st.divider()

# 2. USER INPUTS (Gender and Record ID are removed as requested)
col1, col2 = st.columns(2)

with col1:
    prev_scores = st.number_input("Previous Scores (%)", 0, 100, 77)
    hours_studied = st.number_input("Hours Studied", 0, 24, 5)
    performance_idx = st.number_input("Current Performance Index", 0, 100, 70)

with col2:
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)
    sample_papers = st.number_input("Sample Papers Practiced", 0, 50, 8)
    extra_act = st.selectbox("Extracurricular Activities", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Hidden variables for model compatibility
fixed_value = 101 
fixed_gender = 1

st.divider()

# 3. CALCULATION LOGIC
if st.button("Predict Student Standing", use_container_width=True):
    
    # --- YOUR CUSTOM RULES ---
    # Good if: Score > 50 AND Index > 50 AND Study > 2 AND Sleep < 7
    is_good = (prev_scores > 50 and 
               performance_idx > 50 and 
               hours_studied > 2 and 
               sleep_hours < 7)

    # 4. DATA PREP (For background model check)
    data_dict = {
        "value": [fixed_value],
        "Hours Studied": [hours_studied],
        "Previous Scores": [prev_scores],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers],
        "Performance Index": [performance_idx],
        "Gender_Male": [fixed_gender],
        "Extracurricular Activities_Yes": [extra_act]
    }
    input_df = pd.DataFrame(data_dict)

    try:
        # Reorder to match model
        input_ready = input_df[model.feature_names_in_]
        model.predict(input_ready) # Runs the model silently

        st.subheader("Prediction Result:")

        if is_good:
            # Result shows GOOD
            st.success("✅ Verdict: **GOOD PERFORMANCE**")
            st.balloons()
            st.info("Status: Student is in good standing and meets all requirements.")
        else:
            # Result shows BAD
            st.error("❌ Verdict: **BAD PERFORMANCE**")
            
            # Show exactly which rule was broken
            with st.expander("Show Reasoning:"):
                if prev_scores <= 50: st.write("❌ Score is too low (<= 50%)")
                if performance_idx <= 50: st.write("❌ Performance Index is too low (<= 50)")
                if hours_studied <= 2: st.write("❌ Study hours are insufficient (<= 2)")
                if sleep_hours >= 7: st.write("❌ Sleep hours are too high (>= 7) for this specific rule.")

    except Exception as e:
        st.error(f"Error: {e}")

# 5. FOOTER
st.caption("Developed by Shivam | Nirmala Memorial Foundation College")
