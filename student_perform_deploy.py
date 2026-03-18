import streamlit as st
import joblib
import pandas as pd

# 1. LOAD THE MODEL
# This loads your saved machine learning model
model = joblib.load("best_model.pkl")

# --- UI SETUP ---
st.set_page_config(page_title="Performance Predictor", layout="centered")
st.title("📊 Student Success Predictor")
st.write("Predict student outcome based on academic history.")

st.divider()

# 2. FOCUSED USER INPUTS
# We only show the two features you want to focus on
prev_scores = st.number_input("Overall Previous Scores (%)", 0, 100, 85)
performance_idx = st.number_input("Current Performance Index", 0, 100, 80)

# 3. BACKGROUND DATA (Fixed Values)
# These are required by the model, but we set them to "Good Student" defaults 
# so they don't interfere with your main prediction.
fixed_hours = 12          # Set to a high-average value
fixed_sleep = 8           # Normal sleep
fixed_papers = 5          # Average practice
fixed_value = 100         # Default ID
fixed_gender = 1          # Default Male
fixed_extra = 1           # Default Yes

st.divider()

# 4. PREDICTION LOGIC
if st.button("Analyze Performance", use_container_width=True):
    
    # We combine your inputs with the hidden fixed values
    data_dict = {
        "value": [fixed_value],
        "Hours Studied": [fixed_hours],
        "Previous Scores": [prev_scores],
        "Sleep Hours": [fixed_sleep],
        "Sample Question Papers Practiced": [fixed_papers],
        "Performance Index": [performance_idx],
        "Gender_Male": [fixed_gender],
        "Extracurricular Activities_Yes": [fixed_extra]
    }

    input_df = pd.DataFrame(data_dict)

    try:
        # Reorder to match model's training order
        input_ready = input_df[model.feature_names_in_]
        
        # Get result (0 or 1)
        prediction = model.predict(input_ready)
        result = int(prediction[0])

        st.subheader("Verdict:")
        
        # 0 = GOOD, 1 = BAD (Based on your model's specific training)
        if result == 0:
            st.success(f"✅ Prediction: **GOOD PERFORMANCE**")
            st.balloons()
        else:
            st.error(f"❌ Prediction: **BAD PERFORMANCE**")
            
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.caption("Simplified Predictor | Computer Science Project")
