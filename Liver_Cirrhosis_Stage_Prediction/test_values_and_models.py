#!/usr/bin/env python3
"""
Liver Cirrhosis Stage Prediction - Test Values and Model Validation
==================================================================

This script provides realistic test values for the liver cirrhosis prediction system
and validates that the models work correctly.

Author: Enhanced Liver Cirrhosis Prediction System
Date: September 2025
"""

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")


def load_models():
    """Load all saved models and components"""
    print("🔧 Loading saved models and components...")

    try:
        # Load the best model (XGBoost pipeline)
        best_model = joblib.load("best_liver_cirrhosis_model.pkl")
        print("✅ Best model loaded successfully")

        # Load Random Forest model (for app compatibility)
        rf_model = joblib.load("random_forest_liver_cirrhosis_model.pkl")
        print("✅ Random Forest model loaded successfully")

        # Load feature names
        feature_names = joblib.load("feature_names.pkl")
        print(f"✅ Feature names loaded: {len(feature_names)} features")

        # Load preprocessor
        preprocessor = joblib.load("preprocessor.pkl")
        print("✅ Preprocessor loaded successfully")

        return best_model, rf_model, feature_names, preprocessor

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None, None


def get_test_cases():
    """Generate realistic test cases for liver cirrhosis prediction"""

    print("\n🧪 Creating Test Cases...")
    print("=" * 60)

    test_cases = {
        "Early Stage (Stage 0) - Mild Case": {
            "N_Days": 500,  # Days in study
            "Age": 45,  # Age in years
            "Sex": "F",  # Female
            "Status": "C",  # Alive (Censored)
            "Drug": "Placebo",  # Treatment type
            "Ascites": "N",  # No fluid accumulation
            "Hepatomegaly": "N",  # No liver enlargement
            "Spiders": "N",  # No spider nevi
            "Edema": "N",  # No edema
            "Bilirubin": 1.2,  # Normal-ish bilirubin (mg/dl)
            "Cholesterol": 250,  # Cholesterol (mg/dl)
            "Albumin": 3.8,  # Albumin (g/dl) - normal
            "Copper": 80,  # Copper (μg/day)
            "Alk_Phos": 1200,  # Alkaline phosphatase (U/liter)
            "SGOT": 120,  # AST enzyme (U/ml)
            "Tryglicerides": 150,  # Triglycerides (mg/dl)
            "Platelets": 300,  # Platelet count (per cubic ml/1000)
            "Prothrombin": 11.0,  # Prothrombin time (seconds)
            "Expected_Stage": "Stage 0 (Early)",
        },
        "Moderate Stage (Stage 1) - Progressive Case": {
            "N_Days": 1200,
            "Age": 52,
            "Sex": "M",  # Male
            "Status": "CL",  # Alive with liver transplant
            "Drug": "D-penicillamine",  # Active treatment
            "Ascites": "N",  # No ascites yet
            "Hepatomegaly": "Y",  # Liver enlargement present
            "Spiders": "Y",  # Spider nevi present
            "Edema": "S",  # Slight edema
            "Bilirubin": 2.8,  # Elevated bilirubin
            "Cholesterol": 200,
            "Albumin": 3.2,  # Lower albumin
            "Copper": 150,  # Higher copper
            "Alk_Phos": 1800,  # Elevated alkaline phosphatase
            "SGOT": 180,  # Elevated AST
            "Tryglicerides": 200,
            "Platelets": 250,  # Slightly lower platelets
            "Prothrombin": 12.5,  # Prolonged clotting time
            "Expected_Stage": "Stage 1 (Moderate)",
        },
        "Advanced Stage (Stage 2) - Severe Case": {
            "N_Days": 2000,
            "Age": 58,
            "Sex": "F",
            "Status": "D",  # Deceased
            "Drug": "D-penicillamine",
            "Ascites": "Y",  # Ascites present
            "Hepatomegaly": "Y",  # Liver enlargement
            "Spiders": "Y",  # Spider nevi
            "Edema": "Y",  # Significant edema
            "Bilirubin": 8.5,  # Very high bilirubin
            "Cholesterol": 180,
            "Albumin": 2.5,  # Low albumin (poor synthesis)
            "Copper": 280,  # Very high copper
            "Alk_Phos": 2500,  # Very high alkaline phosphatase
            "SGOT": 250,  # High AST
            "Tryglicerides": 300,
            "Platelets": 150,  # Low platelets (portal hypertension)
            "Prothrombin": 15.0,  # Very prolonged clotting
            "Expected_Stage": "Stage 2 (Advanced)",
        },
        "Borderline Case - Between Stage 0 and 1": {
            "N_Days": 800,
            "Age": 48,
            "Sex": "M",
            "Status": "C",
            "Drug": "Placebo",
            "Ascites": "N",
            "Hepatomegaly": "Y",  # Some enlargement
            "Spiders": "N",
            "Edema": "N",
            "Bilirubin": 2.0,  # Borderline elevated
            "Cholesterol": 220,
            "Albumin": 3.5,  # Borderline low
            "Copper": 110,
            "Alk_Phos": 1500,
            "SGOT": 150,
            "Tryglicerides": 180,
            "Platelets": 280,
            "Prothrombin": 11.8,
            "Expected_Stage": "Stage 0-1 (Borderline)",
        },
        "Young Patient - Early Detection": {
            "N_Days": 300,
            "Age": 35,  # Younger patient
            "Sex": "F",
            "Status": "C",
            "Drug": "D-penicillamine",
            "Ascites": "N",
            "Hepatomegaly": "N",
            "Spiders": "N",
            "Edema": "N",
            "Bilirubin": 0.8,  # Normal bilirubin
            "Cholesterol": 280,  # Higher cholesterol
            "Albumin": 4.0,  # Good albumin
            "Copper": 60,  # Normal copper
            "Alk_Phos": 1000,  # Normal alkaline phosphatase
            "SGOT": 100,  # Normal AST
            "Tryglicerides": 120,
            "Platelets": 350,  # Normal platelets
            "Prothrombin": 10.5,  # Normal clotting
            "Expected_Stage": "Stage 0 (Early/Normal)",
        },
    }

    return test_cases


def convert_to_numeric(test_case):
    """Convert categorical values to numeric for model prediction"""

    # Create a copy to avoid modifying original
    numeric_case = test_case.copy()

    # Mapping dictionaries (same as in the main script)
    mapping_dict = {
        "Status": {"C": 0, "CL": 1, "D": 2},
        "Drug": {"D-penicillamine": 1, "Placebo": 0},
        "Sex": {"M": 1, "F": 0},
        "Ascites": {"Y": 1, "N": 0},
        "Hepatomegaly": {"Y": 1, "N": 0},
        "Spiders": {"Y": 1, "N": 0},
        "Edema": {"Y": 2, "S": 1, "N": 0},
    }

    # Apply mappings
    for feature, mapping in mapping_dict.items():
        if feature in numeric_case:
            numeric_case[feature] = mapping[numeric_case[feature]]

    # Remove expected stage (not a feature)
    if "Expected_Stage" in numeric_case:
        del numeric_case["Expected_Stage"]

    return numeric_case


def test_predictions(model, test_cases, model_name="Model"):
    """Test predictions on all test cases"""

    print(f"\n🔮 Testing {model_name} Predictions...")
    print("=" * 60)

    stage_names = {
        0: "Stage 0 (Early)",
        1: "Stage 1 (Moderate)",
        2: "Stage 2 (Advanced)",
    }

    for case_name, test_case in test_cases.items():
        print(f"\n📋 {case_name}")
        print("-" * 50)

        # Convert to numeric
        numeric_case = convert_to_numeric(test_case)

        # Create DataFrame for prediction
        input_df = pd.DataFrame([numeric_case])

        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            # Display results
            print(f"Expected: {test_case['Expected_Stage']}")
            print(f"Predicted: {stage_names.get(prediction, f'Stage {prediction}')}")
            print(
                f"Confidence: {probabilities[prediction]:.3f} ({probabilities[prediction]*100:.1f}%)"
            )

            # Show all probabilities
            print("All Probabilities:")
            for i, prob in enumerate(probabilities):
                print(
                    f"  {stage_names.get(i, f'Stage {i}')}: {prob:.3f} ({prob*100:.1f}%)"
                )

            # Key indicators
            print(f"Key Indicators:")
            print(f"  • Bilirubin: {test_case['Bilirubin']} mg/dl")
            print(f"  • Albumin: {test_case['Albumin']} g/dl")
            print(f"  • Ascites: {test_case['Ascites']}")
            print(f"  • Age: {test_case['Age']} years")

        except Exception as e:
            print(f"❌ Prediction failed: {e}")

    print("\n" + "=" * 60)


def display_feature_requirements():
    """Display the required features for the GUI application"""

    print("\n📋 GUI Application Input Requirements:")
    print("=" * 60)

    features_info = {
        "Numerical Features": {
            "N_Days": "Number of days in study (e.g., 500-2000)",
            "Age": "Patient age in years (e.g., 30-70)",
            "Bilirubin": "Serum bilirubin in mg/dl (Normal: 0.3-1.2)",
            "Cholesterol": "Cholesterol in mg/dl (e.g., 150-300)",
            "Albumin": "Serum albumin in g/dl (Normal: 3.5-5.0)",
            "Copper": "Urine copper in μg/day (Normal: 15-60)",
            "Alk_Phos": "Alkaline phosphatase in U/liter (Normal: 44-147)",
            "SGOT": "AST enzyme in U/ml (Normal: 8-48)",
            "Tryglicerides": "Triglycerides in mg/dl (Normal: <150)",
            "Platelets": "Platelet count per cubic ml/1000 (Normal: 150-450)",
            "Prothrombin": "Prothrombin time in seconds (Normal: 9.4-12.5)",
        },
        "Categorical Features": {
            "Status": "C=Alive, CL=Liver transplant, D=Deceased",
            "Drug": "D-penicillamine or Placebo",
            "Sex": "M=Male, F=Female",
            "Ascites": "Y=Yes, N=No (fluid in abdomen)",
            "Hepatomegaly": "Y=Yes, N=No (enlarged liver)",
            "Spiders": "Y=Yes, N=No (spider nevi on skin)",
            "Edema": "Y=Yes, S=Slight, N=No (fluid retention)",
        },
    }

    for category, features in features_info.items():
        print(f"\n{category}:")
        for feature, description in features.items():
            print(f"  • {feature}: {description}")


def main():
    """Main testing function"""

    print("🧪 Liver Cirrhosis Stage Prediction - Test Suite")
    print("=" * 60)

    # Load models
    best_model, rf_model, feature_names, preprocessor = load_models()

    if best_model is None:
        print(
            "❌ Could not load models. Make sure you ran liver_cirrhosis_main.py first!"
        )
        return

    # Get test cases
    test_cases = get_test_cases()

    # Test both models
    if best_model:
        test_predictions(best_model, test_cases, "Best Model (XGBoost)")

    if rf_model:
        test_predictions(rf_model, test_cases, "Random Forest Model")

    # Display GUI requirements
    display_feature_requirements()

    print("\n🎯 Quick Test Values for GUI:")
    print("=" * 60)
    print("Copy and paste these values into your PyQt5 app:\n")

    # Display the first test case in a format easy to copy
    first_case = list(test_cases.values())[0]
    print("Early Stage Test Case:")
    for key, value in first_case.items():
        if key != "Expected_Stage":
            print(f"{key}: {value}")

    print(f"\n✅ Test complete! You can now use these values in your PyQt5 app.")
    print(f"Expected prediction: {first_case['Expected_Stage']}")


if __name__ == "__main__":
    main()
