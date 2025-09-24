#!/usr/bin/env python3
"""
Quick Prediction Test for Liver Cirrhosis App
==============================================

This script tests the prediction functionality directly to debug any issues.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

def test_prediction_directly():
    """Test prediction functionality directly"""
    print("🧪 Testing Prediction Functionality Directly")
    print("=" * 50)
    
    try:
        # Load models
        print("📂 Loading models...")
        model = joblib.load("best_liver_cirrhosis_model.pkl")
        print("✅ Model loaded successfully")
        
        # Test data from our previous examples
        test_data = {
            'N_Days': 500,
            'Status': 0,  # C
            'Drug': 0,    # Placebo
            'Age': 45,
            'Sex': 0,     # F
            'Ascites': 0, # N
            'Hepatomegaly': 0, # N
            'Spiders': 0, # N
            'Edema': 0,   # N
            'Bilirubin': 1.2,
            'Cholesterol': 250,
            'Albumin': 3.8,
            'Copper': 80,
            'Alk_Phos': 1200,
            'SGOT': 120,
            'Tryglicerides': 150,
            'Platelets': 300,
            'Prothrombin': 11.0
        }
        
        print("\n📊 Test input data:")
        for key, value in test_data.items():
            print(f"  {key}: {value} ({type(value).__name__})")
        
        # Create DataFrame
        input_df = pd.DataFrame([test_data])
        
        print(f"\n📈 DataFrame info:")
        print(f"  Shape: {input_df.shape}")
        print(f"  Data types:")
        for col, dtype in input_df.dtypes.items():
            print(f"    {col}: {dtype}")
        
        # Check for NaN or infinite values
        print(f"\n🔍 Data validation:")
        print(f"  NaN values: {input_df.isnull().sum().sum()}")
        print(f"  Infinite values: {np.isinf(input_df.select_dtypes(include=[np.number]).values).sum()}")
        
        # Make prediction
        print(f"\n🔮 Making prediction...")
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        stage_names = {0: "Stage 0 (Early)", 1: "Stage 1 (Moderate)", 2: "Stage 2 (Advanced)"}
        
        print(f"✅ Prediction successful!")
        print(f"  Predicted stage: {stage_names.get(prediction, f'Stage {prediction}')}")
        print(f"  Confidence: {probabilities[prediction]:.3f} ({probabilities[prediction]*100:.1f}%)")
        print(f"  All probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"    {stage_names.get(i, f'Stage {i}')}: {prob:.3f} ({prob*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

def test_data_types():
    """Test different data type scenarios"""
    print("\n🔬 Testing Data Type Scenarios")
    print("=" * 50)
    
    # Test mixed data types (what might come from GUI)
    gui_data_mixed = {
        'N_Days': '500',      # String (from QLineEdit)
        'Status': 0,          # Integer (from QComboBox)
        'Drug': '0',          # String (from QComboBox converted)
        'Age': 45.0,          # Float
        'Sex': 0,             # Integer
        'Ascites': '0',       # String
        'Hepatomegaly': 0,    # Integer
        'Spiders': 0,         # Integer
        'Edema': 0,           # Integer
        'Bilirubin': '1.2',   # String
        'Cholesterol': 250,   # Integer
        'Albumin': 3.8,       # Float
        'Copper': '80',       # String
        'Alk_Phos': 1200,     # Integer
        'SGOT': 120,          # Integer
        'Tryglicerides': 150, # Integer
        'Platelets': 300,     # Integer
        'Prothrombin': 11.0   # Float
    }
    
    print("📊 Mixed data types (simulating GUI input):")
    for key, value in gui_data_mixed.items():
        print(f"  {key}: {value} ({type(value).__name__})")
    
    try:
        # Convert to proper types like in the fixed app
        processed_data = []
        numeric_columns = ["N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", 
                         "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]
        categorical_columns = ["Status", "Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
        
        for key in gui_data_mixed.keys():
            value = gui_data_mixed[key]
            if key in numeric_columns:
                # Convert to float
                numeric_value = float(value)
                if np.isnan(numeric_value) or np.isinf(numeric_value):
                    raise ValueError(f"Invalid value for {key}")
                processed_data.append(numeric_value)
            elif key in categorical_columns:
                # Convert to int
                processed_data.append(int(value))
        
        # Create DataFrame with proper types
        input_df = pd.DataFrame([processed_data], columns=list(gui_data_mixed.keys()))
        
        # Set explicit data types
        for col in numeric_columns:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(np.float64)
        
        for col in categorical_columns:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(np.int64)
        
        print(f"\n✅ Data conversion successful!")
        print(f"  Final data types:")
        for col, dtype in input_df.dtypes.items():
            print(f"    {col}: {dtype}")
        
        # Load model and test
        model = joblib.load("best_liver_cirrhosis_model.pkl")
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        stage_names = {0: "Stage 0 (Early)", 1: "Stage 1 (Moderate)", 2: "Stage 2 (Advanced)"}
        print(f"\n✅ Prediction with mixed data types successful!")
        print(f"  Predicted: {stage_names.get(prediction, f'Stage {prediction}')}")
        print(f"  Confidence: {probabilities[prediction]:.1%}")
        
    except Exception as e:
        print(f"❌ Data type conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction_directly()
    test_data_types()
    
    print("\n" + "="*50)
    print("🎯 Test Summary:")
    print("If both tests passed, the prediction functionality should work in the GUI!")
    print("If there were errors, check the error messages above for debugging.")