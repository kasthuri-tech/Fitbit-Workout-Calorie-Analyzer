# ==========================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
from utils import calculate_derived_metrics, preprocess_user_input

# ==========================================
# STEP 2: SETUP THE STREAMLIT WEB PAGE
# ==========================================
# Configure the browser tab title and layout
st.set_page_config(page_title="Fitbit Predictor", layout="wide", page_icon="🔥")

# Add some custom CSS to make buttons look better (letting Streamlit handle Dark/Light mode natively)
st.markdown("""
    <style>
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #ff3333; border-color: #ff3333; }
    h1 { text-align: center; color: #ff4b4b; font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# Display the main titles
st.markdown("<h1>🔥 Fitbit Workout Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #003399;'>Predict Calorie Burn and Analyze Workout Patterns Using Fitbit Data and Machine Learning</h4>", unsafe_allow_html=True)
st.divider()

# ==========================================
# STEP 3: LOAD THE TRAINED MACHINE LEARNING MODELS
# ==========================================
# Use @st.cache_resource so it only loads the heavy models once
@st.cache_resource
def load_models():
    try:
        with gzip.open('fitbit_models.pkl.gz', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

data = load_models()

# If the models fail to load, show an error. Otherwise, proceed.
if data is None:
    st.error("⚠️ Model file 'fitbit_models.pkl.gz' not found! Please ensure the compressed model file is present.")
else:
    # Extract the necessary tools from our saved data
    scaler = data['scaler']
    le_dict = data['label_encoders']
    models = data['models']
    feature_names = data['features']
    
    # ==========================================
    # STEP 4: CREATE THE SIDEBAR MENU (MODEL SELECTION)
    # ==========================================
    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("As requested, showcasing all 8 models. If your system is slow, avoid Random Forest.")
    
    # Create a dropdown box containing all 8 machine learning models
    model_choice = st.sidebar.selectbox("Choose Predictive Model:", list(models.keys()))
    
    st.sidebar.info("💡 **Unsupervised Learning Note:** Clustering (KMeans, DBSCAN, Hierarchical) was performed in the backend to find hidden workout segments.")
    
    # ==========================================
    # STEP 5: CREATE THE USER INPUT COLUMNS
    # ==========================================
    # Split the screen into 3 neat columns for easy user input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 User Profile")
        age = st.number_input("Age", 15, 80, 30)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
        height = st.number_input("Height (m)", 1.4, 2.2, 1.75)
        fat_perc = st.number_input("Body Fat Percentage (%)", 5.0, 40.0, 20.0)
        experience = st.selectbox("Experience Level", ['Beginner', 'Intermediate', 'Advanced'])
        
    with col2:
        st.markdown("### ❤️ Heart Rate Data")
        resting_bpm = st.number_input("Resting BPM", 40, 100, 60)
        avg_bpm = st.number_input("Average BPM", 80, 180, 120)
        max_bpm = st.number_input("Maximum BPM", 100, 220, 150)
        
    with col3:
        st.markdown("### 🏃 Workout Details")
        # Automatically pull workout types from the encoder, or use defaults
        workout_type_classes = list(le_dict['Workout_Type'].classes_) if 'Workout_Type' in le_dict else ['Cardio', 'Strength', 'Yoga', 'HIIT', 'Running', 'Cycling']
        workout_type = st.selectbox("Workout Type", workout_type_classes)
        
        duration = st.number_input("Session Duration (hours)", 0.1, 4.0, 1.0)
        water = st.number_input("Water Intake (liters)", 0.0, 5.0, 1.5)
        freq = st.number_input("Workout Frequency (days/week)", 1, 7, 3)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================
    # STEP 6: PERFORM THE PREDICTION LOGIC
    # ==========================================
    # When the user clicks the predict button
    if st.button("🚀 Predict Calories Burned", use_container_width=True):
        
        # Step 6a: Calculate the mathematical derived features cleanly using reusable utility function
        bmi, base_met, hr_intensity, effective_met = calculate_derived_metrics(
            weight, height, avg_bpm, resting_bpm, max_bpm
        )
        
        # Step 6b: Map strings back to numbers if needed (Experience Level)
        experience_map = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
        
        # Step 6c: Store everything in a dictionary exactly matching our dataset
        input_dict = {
            'Age': age,
            'Gender': gender,
            'Weight (kg)': weight,
            'Height (m)': height,
            'Max_BPM': max_bpm,
            'Avg_BPM': avg_bpm,
            'Resting_BPM': resting_bpm,
            'Session_Duration (hours)': duration,
            'Workout_Type': workout_type,
            'Fat_Percentage': fat_perc,
            'Water_Intake (liters)': water,
            'Workout_Frequency (days/week)': freq,
            'Experience_Level': experience_map[experience],
            'BMI': bmi,
            'Base_MET': base_met,
            'HR_Intensity': hr_intensity,
            'Effective_MET': effective_met
        }
        
        # Step 6d & 6e: Transform, encode, and scale input array using our fully modular preprocessing function
        input_scaled = preprocess_user_input(input_dict, le_dict, feature_names, scaler)
        
        # Step 6f: Make the prediction using the selected model
        selected_model = models[model_choice]
        prediction = selected_model.predict(input_scaled)[0]
        
        # ==========================================
        # STEP 7: SHOW THE RESULT TO THE USER
        # ==========================================
        st.success(f"### 🎉 Predicted Calories Burned: **{prediction:.2f} kcal**")
        st.balloons()
