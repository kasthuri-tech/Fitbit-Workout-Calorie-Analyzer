# ==========================================
# REUSABLE UTILITY FUNCTIONS (utils.py)
# Demonstrates Code Reusability and Professional Modularity
# ==========================================
import pandas as pd

def calculate_derived_metrics(weight, height, avg_bpm, resting_bpm, max_bpm):
    # Step 1: Reusable function to calculate mathematical fitness indicators.
    # Can be imported and reused by any mobile app or backend service.
    bmi = weight / (height ** 2)
    base_met = 1.2  # Approximation for resting metabolic equivalent
    hr_intensity = (avg_bpm - resting_bpm) / (max_bpm - resting_bpm + 0.0001)
    effective_met = base_met + (hr_intensity * 10)
    return bmi, base_met, hr_intensity, effective_met

def preprocess_user_input(input_dict, le_dict, feature_names, scaler):
    # Step 2: Reusable function to transform raw input dict into fully scaled machine learning ready input array.
    # Ensures zero logic duplication across multiple inference platforms.
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Encode categorical columns securely using saved encoders
    for col in le_dict.keys():
        if col in input_df.columns:
            try:
                input_df[col] = le_dict[col].transform(input_df[col])
            except ValueError:
                input_df[col] = 0  # Fallback for unexpected string inputs
                
    # Force exact column ordering expected by ML models
    input_df = input_df[feature_names]
    
    # Apply standard scaling transformation
    input_scaled = scaler.transform(input_df)
    return input_scaled
