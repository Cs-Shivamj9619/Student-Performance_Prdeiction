import streamlit as st  # For creating the website interface
import joblib           # For loading your 'best_model.pkl' file
import pandas as pd     # For organizing the data into a table (DataFrame)

# 1. LOAD THE MODEL
# This line finds your saved machine learning model and prepares it for use
model = joblib.load("best_model.pkl")

# 2. APP UI SETUP
# These lines create the title and the headings on your website
st.set_page_config(page_title="Student Performance Pro")
st.title("🎓 Student Performance Prediction")
st.write("Enter the details below to predict if the student's performance is Good or Bad.")

st.divider() # Adds a horizontal line for better design

# 3. USER INPUT FIELDS
# We create 8 input boxes because your model expects exactly 8 pieces of data
col1, col2 = st.columns(2) # Splits the screen into two columns for a cleaner look

with col1:
    value_idx = st.number_input("Record ID", value=100) # 'value' column
    hours_studied = st.number_input("Hours Studied", 0, 24, 15) # 'Hours Studied'
    prev_scores = st.number_input("Previous Scores (%)", 0, 100, 90) # 'Previous Scores'
    sleep_hours = st.number_input("Sleep Hours", 0, 24, 8) # 'Sleep Hours'

with col2:
    sample_papers = st.number_input("Papers Practiced", 0, 50, 10) # 'Sample Question Papers'
    performance_idx = st.number_input("Performance Index", 0, 100, 85) # 'Performance Index'
    gender = st.selectbox("Gender", [1, 0]) # 1 for Male, 0 for Female
    extra_act = st.selectbox("Extracurriculars", [1, 0]) # 1 for Yes, 0 for No

st.divider()

# 4. PREDICTION LOGIC
# This code runs only when the 'Predict' button is clicked
if st.button("Predict Result", use_container_width=True):
    
    # Store all user inputs in a dictionary with the exact names the model wants
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

    # Convert the dictionary into a DataFrame (the format ML models prefer)
    input_df = pd.DataFrame(data_dict)

    # Use 'feature_names_in_' to automatically reorder columns to match the model
    input_ready = input_df[model.feature_names_in_]
    
    # Send the data to the model to get a result (0 or 1)
    prediction = model.predict(input_ready)
    result = prediction[0]

    st.subheader("Final Verdict:")
    
    # --- LOGIC CORRECTION ---
    # In your model, 0 = GOOD and 1 = BAD. 
    # We use st.success for Good and st.error for Bad to color the results.
    if result == 0:
        st.success(f"✅ Outcome: **GOOD PERFORMANCE** (Model Output: {result})")
    else:
        st.error(f"❌ Outcome: **BAD PERFORMANCE** (Model Output: {result})")

# FOOTER
st.caption("Developed by Shivam | B.Sc. Computer Science Project")
