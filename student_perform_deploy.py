import streamlit as st
import joblib
import pandas as pd

# 1. LOAD THE MODEL
# We assume this is a Regressor model (predicts a number, not just 0 or 1)
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Performance Predictor", layout="wide")
st.title("📈 Student Performance Index Predictor")
st.write("Enter the metrics below to predict the estimated Performance Index.")

st.divider()

# 2. USER INPUTS
col1, col2, col3 = st.columns(3)

with col1:
    prev_scores = st.number_input("Previous Scores (%)", 0, 100, 80)
    hours_studied = st.number_input("Hours Studied", 0, 24, 8)

with col2:
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)
    sample_papers = st.number_input("Papers Practiced", 0, 50, 5)

with col3:
    extra_act = st.selectbox("Extracurriculars", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# 3. BACKGROUND DATA (Hidden from User)
fixed_value = 100 
fixed_gender = 1

st.divider()

# 4. PREDICTION LOGIC
if st.button("Calculate Performance Index", use_container_width=True):
    
    # Organize data for the model
    data_dict = {
        "value": [fixed_value],
        "Hours Studied": [hours_studied],
        "Previous Scores": [prev_scores],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers],
        "Performance Index": [0], # Placeholder if the model requires it in the input array
        "Gender_Male": [fixed_gender],
        "Extracurricular Activities_Yes": [extra_act]
    }

    input_df = pd.DataFrame(data_dict)

    try:
        # Match the model's expected column order
        input_ready = input_df[model.feature_names_in_]
        
        # Predict the numerical Index
        prediction = model.predict(input_ready)
        final_index = round(float(prediction[0]), 2)

        # 5. DISPLAY RESULTS
        st.subheader("Predicted Result:")
        st.metric(label="Estimated Performance Index", value=f"{final_index}%")
        
        # Simple progress bar for visual impact
        st.progress(min(final_index / 100, 1.0))

        if final_index >= 75:
            st.success("This indicates a strong academic standing!")
        elif final_index >= 50:
            st.warning("This indicates average performance. Consider increasing study hours.")
        else:
            st.error("This indicates a risk of poor performance.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# Footer
st.caption("Developed by Shivam | B.Sc. Computer Science | Final Goal Version")
