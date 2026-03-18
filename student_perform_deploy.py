import streamlit as st
import joblib
import pandas as pd

# 1. LOAD THE MODEL
# This loads your saved model (Regression model)
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Performance Index Predictor", layout="centered")
st.title("📈 Student Performance Index Calculator")
st.write("Enter student data to calculate the specific Performance Index (10-100).")

st.divider()

# 2. INPUT FIELDS 
# Removed Gender and Record ID as requested.
# I have set the defaults to match a typical average student.
col1, col2 = st.columns(2)

with col1:
    prev_scores = st.number_input("Previous Scores (%)", 0, 100, 70)
    hours_studied = st.number_input("Hours Studied", 0, 24, 5)
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)

with col2:
    sample_papers = st.number_input("Sample Papers Practiced", 0, 50, 5)
    # Using 'Extracurricular Activities' as it's a standard part of the calculation
    extra_act = st.selectbox("Extracurricular Activities", options=[1, 0], 
                             format_func=lambda x: "Yes" if x == 1 else "No")

# 3. HIDDEN DATA 
# These must be sent to the model so it doesn't crash, but the user won't see them.
# We use standard 'neutral' values.
hidden_value = 100
hidden_gender = 1 
# Note: If your model requires 'Performance Index' as an input to predict a 
# 'Good/Bad' label, this code flips it to show you the Index instead.

st.divider()

# 4. PREDICTION LOGIC
if st.button("Calculate Performance Index", use_container_width=True):
    
    # Construct the data dictionary with all 8 expected fields
    data_dict = {
        "value": [hidden_value],
        "Hours Studied": [hours_studied],
        "Previous Scores": [prev_scores],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers],
        "Performance Index": [0], # Placeholder: the model will calculate the real one
        "Gender_Male": [hidden_gender],
        "Extracurricular Activities_Yes": [extra_act]
    }

    input_df = pd.DataFrame(data_dict)

    try:
        # Reorder columns to match the model training order
        input_ready = input_df[model.feature_names_in_]
        
        # Predict the Index
        prediction = model.predict(input_ready)
        final_index = round(float(prediction[0]), 2)

        # 5. DISPLAY RESULTS
        st.subheader("Predicted Performance Index:")
        st.metric(label="Score Out of 100", value=f"{final_index}%")
        
        # Visual feedback based on the number
        if final_index >= 80:
            st.success("Excellent standing! The student is in the top tier.")
        elif final_index >= 50:
            st.info("Average standing. Performance is stable.")
        else:
            st.warning("Low standing. Academic intervention may be required.")

    except Exception as e:
        st.error(f"Calculation Error: {e}")

# Footer
st.caption("Developed by Shivam | B.Sc. Computer Science")
