import streamlit as st  # For the website interface
import joblib           # To load the .pkl model file
import pandas as pd     # To organize the data for the model

# 1. LOAD THE MODEL
# This loads your trained machine learning model
model = joblib.load("best_model.pkl")

# --- UI DESIGN ---
st.set_page_config(page_title="Student Performance Pro")
st.title("🎓 Student Performance Prediction")
st.write("Enter student data to predict if their performance is Good or Bad.")

st.divider()

# 2. INPUT FIELDS (Exactly 8 features)
col1, col2 = st.columns(2)

with col1:
    value_idx = st.number_input("Record ID (Value)", value=100)
    hours_studied = st.number_input("Hours Studied", 0, 24, 15) # Default set to 15
    prev_scores = st.number_input("Previous Scores (%)", 0, 100, 90) # Default set to 90
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 8)

with col2:
    sample_papers = st.number_input("Papers Practiced", 0, 50, 10)
    performance_idx = st.number_input("Performance Index", 0, 100, 85) # Default set to 85
    gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    extra_act = st.selectbox("Extracurriculars", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# 3. PREDICTION LOGIC
if st.button("Predict Result", use_container_width=True):
    
    # Store inputs in a dictionary using the exact names the model expects
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

    # Convert to DataFrame
    input_df = pd.DataFrame(data_dict)

    try:
        # Reorder columns to match the model's training order
        input_ready = input_df[model.feature_names_in_]
        
        # Make the prediction
        prediction = model.predict(input_ready)
        result = prediction[0]

        st.subheader("Final Verdict:")
        
        # LOGIC: If Output is 0, it means GOOD. If 1, it means BAD.
        if result == 0:
            st.success(f"✅ Outcome: **GOOD PERFORMANCE** (Model Output: {result})")
            st.balloons() # Nice effect for a good result!
        else:
            st.error(f"❌ Outcome: **BAD PERFORMANCE** (Model Output: {result})")
            st.info("Tip: Try increasing 'Hours Studied' or 'Performance Index' to see a GOOD result.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# FOOTER
st.divider()
st.caption("Developed by Shivam | B.Sc. Computer Science Project")
