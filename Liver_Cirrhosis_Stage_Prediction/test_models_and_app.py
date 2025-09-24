#!/usr/bin/env python3
"""
Test script for Liver Cirrhosis Prediction Models
Tests both the saved models and provides sample data for GUI testing
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")


def load_and_test_models():
    """Load and test all available models"""
    print("🧪 LIVER CIRRHOSIS MODEL & APP TESTING")
    print("=" * 60)

    # Test model loading
    model_files = [
        "sklearn_only_liver_cirrhosis_model.pkl",
        "best_liver_cirrhosis_model.pkl",
        "random_forest_liver_cirrhosis_model.pkl",
    ]

    models_loaded = {}

    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                models_loaded[model_file] = model
                print(f"✅ Successfully loaded: {model_file}")
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
        else:
            print(f"⚠️  File not found: {model_file}")

    # Test preprocessing components
    print(f"\n🔧 PREPROCESSING COMPONENTS:")
    components = [
        "preprocessor.pkl",
        "sklearn_preprocessor.pkl",
        "scaler_liver_cirrhosis.pkl",
        "feature_names.pkl",
        "feature_names_sklearn.pkl",
    ]

    loaded_components = {}
    for component in components:
        if os.path.exists(component):
            try:
                loaded_components[component] = joblib.load(component)
                print(f"✅ Loaded: {component}")
            except Exception as e:
                print(f"❌ Failed to load {component}: {e}")
        else:
            print(f"⚠️  Not found: {component}")

    return models_loaded, loaded_components


def generate_test_cases():
    """Generate realistic test cases for manual GUI testing"""
    print(f"\n🎯 SAMPLE TEST CASES FOR GUI TESTING")
    print("=" * 60)

    # Test Case 1: Early Stage (Stage 1)
    test_case_1 = {
        "N_Days": 200,
        "Status": 0,  # Censored
        "Drug": 1,  # D-penicillamine
        "Age": 45.5,
        "Sex": 0,  # Female
        "Ascites": 0,  # No
        "Hepatomegaly": 0,  # No
        "Spiders": 0,  # No
        "Edema": 0,  # No Edema
        "Bilirubin": 1.2,
        "Cholesterol": 280,
        "Albumin": 4.1,
        "Copper": 45,
        "Alk_Phos": 1200,
        "SGOT": 45,
        "Tryglicerides": 130,
        "Platelets": 350,
        "Prothrombin": 11.5,
    }

    # Test Case 2: Moderate Stage (Stage 2)
    test_case_2 = {
        "N_Days": 800,
        "Status": 1,  # Censored due to Liver TX
        "Drug": 0,  # Placebo
        "Age": 55.2,
        "Sex": 1,  # Male
        "Ascites": 1,  # Yes
        "Hepatomegaly": 1,  # Yes
        "Spiders": 0,  # No
        "Edema": 1,  # Edema not present with Diuretics
        "Bilirubin": 4.5,
        "Cholesterol": 200,
        "Albumin": 3.2,
        "Copper": 120,
        "Alk_Phos": 1800,
        "SGOT": 95,
        "Tryglicerides": 180,
        "Platelets": 220,
        "Prothrombin": 13.8,
    }

    # Test Case 3: Advanced Stage (Stage 3)
    test_case_3 = {
        "N_Days": 1500,
        "Status": 2,  # Death
        "Drug": 1,  # D-penicillamine
        "Age": 65.8,
        "Sex": 0,  # Female
        "Ascites": 1,  # Yes
        "Hepatomegaly": 1,  # Yes
        "Spiders": 1,  # Yes
        "Edema": 2,  # Edema Present
        "Bilirubin": 12.5,
        "Cholesterol": 150,
        "Albumin": 2.1,
        "Copper": 250,
        "Alk_Phos": 3500,
        "SGOT": 180,
        "Tryglicerides": 220,
        "Platelets": 120,
        "Prothrombin": 16.2,
    }

    test_cases = [
        ("Test Case 1 - Expected: Early Stage (Stage 1)", test_case_1),
        ("Test Case 2 - Expected: Moderate Stage (Stage 2)", test_case_2),
        ("Test Case 3 - Expected: Advanced Stage (Stage 3)", test_case_3),
    ]

    for name, case in test_cases:
        print(f"\n📝 {name}")
        print("-" * 50)
        for key, value in case.items():
            print(f"{key}: {value}")

    return test_cases


def test_model_predictions(models_loaded, test_cases):
    """Test model predictions with sample cases"""
    print(f"\n🔍 AUTOMATED MODEL TESTING")
    print("=" * 60)

    if not models_loaded:
        print("❌ No models loaded for testing")
        return

    # Use the best available model for testing
    model_name = list(models_loaded.keys())[0]
    model = models_loaded[model_name]

    print(f"🧠 Testing with: {model_name}")

    # Define feature names in correct order
    feature_names = [
        "N_Days",
        "Status",
        "Drug",
        "Age",
        "Sex",
        "Ascites",
        "Hepatomegaly",
        "Spiders",
        "Edema",
        "Bilirubin",
        "Cholesterol",
        "Albumin",
        "Copper",
        "Alk_Phos",
        "SGOT",
        "Tryglicerides",
        "Platelets",
        "Prothrombin",
    ]

    stage_names = {
        0: "Stage 1 (Early)",
        1: "Stage 2 (Moderate)",
        2: "Stage 3 (Advanced)",
    }

    for case_name, case_data in test_cases:
        print(f"\n🎯 Testing: {case_name}")
        try:
            # Convert to DataFrame
            input_data = [case_data[feature] for feature in feature_names]
            input_df = pd.DataFrame([input_data], columns=feature_names)

            # Make prediction
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            predicted_stage = stage_names.get(prediction, f"Stage {prediction}")
            confidence = probabilities[prediction]

            print(f"   🎯 Prediction: {predicted_stage}")
            print(f"   🎲 Confidence: {confidence:.1%}")
            print(f"   📊 All Probabilities:")
            for i, prob in enumerate(probabilities):
                stage = stage_names.get(i, f"Stage {i}")
                print(f"      • {stage}: {prob:.1%}")

        except Exception as e:
            print(f"   ❌ Error during prediction: {e}")


def check_app_compatibility():
    """Check if app files are compatible with current models"""
    print(f"\n🖥️  GUI APPLICATION COMPATIBILITY CHECK")
    print("=" * 60)

    # Check if app file exists
    app_file = "liver_cirrhosis_app.py"
    if not os.path.exists(app_file):
        print("❌ GUI application file not found")
        return False

    print("✅ GUI application file found")

    # Check dataset
    if os.path.exists("liver_cirrhosis.csv"):
        print("✅ Dataset file available")
    else:
        print("❌ Dataset file missing")
        return False

    # Check model files that GUI expects
    expected_files = [
        "random_forest_liver_cirrhosis_model.pkl",
        "preprocessor.pkl",
        "scaler_liver_cirrhosis.pkl",
    ]

    for file in expected_files:
        if os.path.exists(file):
            print(f"✅ {file} - Available")
        else:
            print(f"⚠️  {file} - Missing (GUI may use fallback)")

    print(f"\n💡 GUI Testing Instructions:")
    print(f"1. The GUI should be running now")
    print(f"2. Use the test cases above to fill in the form")
    print(f"3. Click 'Predict Cirrhosis Stage' to test predictions")
    print(f"4. Try the 'Model Comparison' tab for benchmarking")
    print(f"5. Check 'Model Insights' tab for feature importance")
    print(f"6. Test 'External Validation' for advanced features")

    return True


def main():
    """Main testing function"""
    # Test model loading
    models_loaded, components_loaded = load_and_test_models()

    # Generate test cases
    test_cases = generate_test_cases()

    # Test model predictions
    test_model_predictions(models_loaded, test_cases)

    # Check app compatibility
    check_app_compatibility()

    print(f"\n🎉 TESTING COMPLETED!")
    print(f"💡 Use the test cases above in the GUI to validate predictions")
    print(f"📱 The PyQt5 app should be running in a separate window")


if __name__ == "__main__":
    main()
