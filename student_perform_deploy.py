import streamlit as st
import joblib
import pandas as pd

# 1. Load the model
model = joblib.load("best_model.pkl")

st.title("Student Performance Prediction")

# 2. Map every possible feature name (just to be safe)
# The order here doesn't matter; the dictionary will organize them.
inputs = {}
inputs['Hours Studied'] = st.number_input("Hours Studied", 0, 24, 10)
inputs['Previous Scores'] = st.number_input("Previous Scores", 0, 100, 85)
inputs['Sleep Hours'] = st.number_input("Sleep Hours", 0, 24, 8)
inputs['Sample Question Papers Practiced'] = st.number_input("Papers Practiced", 0, 50, 10)
inputs['Tuition'] = st.selectbox("Tuition (0=No, 1=Yes)", [0, 1])
inputs['Extracurricular Activities'] = st.selectbox("Extracurricular (0=No, 1=Yes)", [0, 1])
inputs['Parental Education Level'] = st.selectbox("Parental Education (0=Low, 1=High)", [0, 1])

# 3. Predict Button
if st.button("Predict Result"):
    # Create a DataFrame from the inputs
    input_df = pd.DataFrame([inputs])

    try:
        # MAGIC FIX: This reorders the columns to match the model EXACTLY
        # using the names the model saved during training.
        input_data_reordered = input_df[model.feature_names_in_]
        
        prediction = model.predict(input_data_reordered)
        result = prediction[0]

        st.divider()
        st.subheader("Final Result:")

        # Sometimes models predict a number (like a score) instead of a label.
        # If your result is a number > 50, it's 'GOOD'.
        if result == 1 or (isinstance(result, (int, float)) and result > 50):
            st.success(f"Performance: GOOD (Score/Value: {result})")
        else:
            st.error(f"Performance: BAD (Score/Value: {result})")
            
    except Exception as e:
        # If the names don't match, this will tell us the CORRECT names.
        st.error("Column Name Mismatch!")
        st.write("The model actually wants these names:", model.feature_names_in_)
