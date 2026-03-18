import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="Performance Predictor", page_icon="🎓")

# --- LOAD THE PRE-TRAINED MODEL ---
@st.cache_resource
def load_my_model():
    # This loads the .pkl file shown in your screenshot
    return joblib.load('best_model.pkl')

model = load_my_model()

# --- USER INTERFACE ---
st.title("📈 Student Performance Index Predictor")
st.write("Enter student data below to get an instant performance prediction.")

# Form for user input
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        hours = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
        prev_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
        sleep = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
        
    with col2:
        extra_classes = st.number_input("Extra Classes Attended", min_value=0, max_value=10, value=0)
        # Note: If your model was trained with 'Yes/No' (0/1), 
        # make sure these features match your training columns exactly!

    submit = st.form_submit_button("Predict Performance Index")

# --- PREDICTION LOGIC ---
if submit:
    # 1. Arrange features in the exact order your model expects
    # (Update this list if your model has more/different columns!)
    feature_list = np.array([[hours, prev_scores, sleep, extra_classes]])
    
    # 2. Predict
    prediction = model.predict(feature_list)
    
    # 3. Display result
    st.divider()
    st.subheader(f"Predicted Performance Index: {prediction[0]:.2f}")
    
    # Simple advice based on score
    if prediction[0] > 75:
        st.success("Keep it up! The student is on a great track.")
    elif prediction[0] > 40:
        st.info("Stable performance. Consistency is key.")
    else:
        st.warning("Low prediction. Suggesting additional support or revision.")
