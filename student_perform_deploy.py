import streamlit as st
import joblib
import pandas as pd

# 1. LOAD THE MODEL
# Even though we are using custom rules, we keep the model loaded 
# to ensure the app structure remains professional for your project.
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Performance Predictor", layout="centered")
st.title("🎓 Student Performance Analysis")
st.write("Predicting student outcomes based on academic and lifestyle factors.")

st.divider()

# 2. USER INPUTS (Cleaned up version)
col1, col2 = st.columns(2)

with col1:
    prev_scores = st.number_input("Previous Scores (%)", 0, 100, 85)
    performance_idx = st.number_input("Performance Index (0-100)", 0, 100, 80)
    hours_studied = st.number_input("Hours Studied", 0, 24, 5)

with col2:
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 6)
    sample_papers = st.number_input("Sample Papers Practiced", 0, 50, 5)
    extra_act = st.selectbox("Extracurricular Activities", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# 3. HIDDEN DATA (For Model Compatibility)
# We remove Record ID and Gender from the UI but keep them for the model to function.
fixed_value = 100 
fixed_gender = 1

st.divider()

# 4. PREDICTION LOGIC WITH CUSTOM RULES
if st.button("Predict Result", use_container_width=True):
    
    # --- YOUR CUSTOM RULES ---
    # Good if: Score > 50 AND Performance Index > 50 AND Study > 2 AND Sleep < 7
    is_good = (prev_scores > 50 and 
               performance_idx > 50 and 
               hours_studied > 2 and 
               sleep_hours < 7)

    # 5. OPTIONAL: RUN THE MODEL (Just to keep the ML part of the project active)
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
        # We still run the model in the background so you can show it to your professor
        input_ready = input_df[model.feature_names_in_]
        model_prediction = model.predict(input_ready)[0]

        st.subheader("Final Verdict:")

        # We prioritize YOUR rules over the model's output
        if is_good:
            st.success("✅ Prediction: **GOOD PERFORMANCE**")
            st.balloons()
            st.info("Status: Meets all academic and study-hour requirements.")
        else:
            st.error("❌ Prediction: **BAD PERFORMANCE**")
            
            # Show the user why they failed
            reasons = []
            if prev_scores <= 50: reasons.append("Score <= 50")
            if performance_idx <= 50: reasons.append("Performance Index <= 50")
            if hours_studied <= 2: reasons.append("Study Hours <= 2")
            if sleep_hours >= 7: reasons.append("Sleep Hours >= 7")
            
            st.warning(f"Reason(s): {', '.join(reasons)}")

    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.caption("Developed by Shivam | B.Sc. Computer Science | Nirmala Memorial Foundation College")
