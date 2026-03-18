import streamlit as st  # Web framework
import joblib           # To load your model
import pandas as pd     # To handle data

# 1. LOAD THE MODEL
model = joblib.load("best_model.pkl")

# --- UI SETUP ---
st.set_page_config(page_title="Academic Performance Predictor")
st.title("🎓 Student Performance Analysis")
st.write("Enter details below. Note: Academic scores are the primary priority for this prediction.")

st.divider()

# 2. INPUT FIELDS
# We organize them so 'Scores' and 'Index' are at the very top (Priority)
prev_scores = st.number_input("Previous Scores (%) - HIGH PRIORITY", 0, 100, 85)
performance_idx = st.number_input("Performance Index - HIGH PRIORITY", 0, 100, 80)

st.subheader("Additional Factors (Secondary)")
col1, col2 = st.columns(2)

with col1:
    # We set these defaults to "Good" values so they don't drag the score down
    hours_studied = st.number_input("Hours Studied", 0, 24, 12) 
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 8)
    value_idx = st.number_input("Record ID", value=100)

with col2:
    sample_papers = st.number_input("Papers Practiced", 0, 50, 5)
    gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    extra_act = st.selectbox("Extracurriculars", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# 3. PREDICTION LOGIC
if st.button("Predict Outcome", use_container_width=True):
    
    # Map all inputs to the exact names the model requires
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
        # Reorder columns to match the model training order
        input_ready = input_df[model.feature_names_in_]
        
        # Get result
        prediction = model.predict(input_ready)
        result = int(prediction[0])

        st.subheader("Result:")
        
        # 0 = GOOD, 1 = BAD (As per your model's behavior)
        if result == 0:
            st.success(f"✅ Predicted Result: **GOOD PERFORMANCE**")
            st.balloons()
        else:
            st.error(f"❌ Predicted Result: **BAD PERFORMANCE**")
            st.info("The model suggests that despite the scores, other factors or a very low Performance Index are impacting the result.")
            
    except Exception as e:
        st.error(f"Error: {e}")

# 4. FOOTER
st.caption("Developed by Shivam | Nirmala Memorial Foundation College Project")
