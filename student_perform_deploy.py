import streamlit as st  # Import Streamlit for the web interface
import joblib           # Import Joblib to load your saved ML model
import pandas as pd     # Import Pandas to handle the data in a table format

# 1. Load the trained model file
model = joblib.load("best_model.pkl") # Loads the .pkl file you created

# 2. Set the Title of your Web App
st.title("Student Performance Checker") # Displays a big heading on the page

# 3. Create Input Boxes for the 8 features the model needs
val = st.number_input("Record ID", value=100) # Input for 'value' column
hrs = st.number_input("Hours Studied", 0, 24, 15) # Input for 'Hours Studied'
prev = st.number_input("Previous Score", 0, 100, 90) # Input for 'Previous Scores'
sleep = st.number_input("Sleep Hours", 0, 24, 8) # Input for 'Sleep Hours'
paper = st.number_input("Papers Practiced", 0, 50, 10) # Input for 'Sample Papers'
p_idx = st.number_input("Performance Index", 0, 100, 85) # Input for 'Performance Index'
gen = st.selectbox("Gender (1=Male, 0=Female)", [1, 0]) # Input for 'Gender_Male'
ext = st.selectbox("Extracurricular (1=Yes, 0=No)", [1, 0]) # Input for 'Extracurricular_Yes'

# 4. Create the 'Predict' Button
if st.button("Predict"): # When the button is clicked, do the following:

    # Put all inputs into a dictionary (Label : Value)
    # The labels MUST match the names your model was trained on
    data = {
        "value": [val],
        "Hours Studied": [hrs],
        "Previous Scores": [prev],
        "Sleep Hours": [sleep],
        "Sample Question Papers Practiced": [paper],
        "Performance Index": [p_idx],
        "Gender_Male": [gen],
        "Extracurricular Activities_Yes": [ext]
    }

    # Convert the dictionary into a Pandas DataFrame
    df = pd.DataFrame(data)

    # Reorder the columns to match the model's training order exactly
    df = df[model.feature_names_in_]

    # Use the model to make a prediction (returns 0 or 1)
    prediction = model.predict(df)
    result = prediction[0] # Get the first (and only) result from the list

    # 5. Display the result to the user
    # Based on your model output, 0 means 'Good'
    if result == 0:
        st.success("Result: GOOD PERFORMANCE") # Show a green box for Good
    else:
        st.error("Result: BAD PERFORMANCE") # Show a red box for Bad
