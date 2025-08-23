import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
import XGBoost
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight

import warnings

warnings.filterwarnings("ignore")


FILE_PATH = "Lung Cancer\dataset_med.csv"

df = pd.read_csv(FILE_PATH)
print(f"Original Dataset Shape: {df.shape}")
print("Data Loaded Successfully")

print(f"\nDataset Info:")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")

# Drop unnecessary columns
print(f"\nDropping unnecessary columns...")
columns_to_drop = ["id", "country", "diagnosis_date", "end_treatment_date"]
df_clean = df.drop(columns=columns_to_drop)
print(f"Shape after dropping columns: {df_clean.shape}")

# Check for missing values
print(f"\nMissing Values Check:")
missing_values = df.isnull().sum()
print(missing_values)
total_missing = missing_values.sum()
print(f"Total missing values: {total_missing}")

# Remove rows with any missing values
print(f"\nHandling missing values...")
print(f"Rows before removing missing values: {len(df_clean)}")
df_clean.dropna(inplace=True)
print(f"Rows after removing missing values: {len(df_clean)}")
print(f"Final dataset shape: {df_clean.shape}")

# Check the target variable distribution
print(f"\nTarget Variable Analysis (Survival):")
target_dist = df_clean["survived"].value_counts()
target_pct = df_clean["survived"].value_counts(normalize=True) * 100
print(f"Not Survived (0): {target_dist[0]} samples ({target_pct[0]:.1f}%)")
print(f"Survived (1): {target_dist[1]} samples ({target_pct[1]:.1f}%)")
print(f"Class imbalance ratio: {target_pct[0]/target_pct[1]:.2f}:1")

# print("Categorical column values:")
# for col in categorical_columns:
#    print(f"\n{col}:")
#    value_counts = df_clean[col].value_counts()
#    for value, count in value_counts.items():
#        print(f"  - {value}: {count} ({count/len(df_clean)*100:.1f}%)")

# Class Mapping for Easier UI Allocation
df_clean["gender"] = df_clean["gender"].map({"Male": 1, "Female": 0})

df_clean["family_history"] = df_clean["family_history"].map({"No": 0, "Yes": 1})

df_clean["cancer_stage"] = df_clean["cancer_stage"].map(
    {"Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3}
)

df_clean["treatment_type"] = df_clean["treatment_type"].map(
    {"Surgery": 0, "Chemotherapy": 1, "Radiation": 2, "Combined": 3}
)

df_clean["smoking_status"] = df_clean["smoking_status"].map(
    {"Never Smoked": 0, "Passive Smoker": 1, "Former Smoker": 2, "Current Smoker": 3}
)

# Prepare features and target
X = df_clean.drop("survived", axis=1)
y = df_clean["survived"]

print(f"\nFeature columns: {list(X.columns)}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData Split:")
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# Pipeline setup

numerical_features = ["age", "bmi", "cholesterol_level"]
categorical_features = [
    "gender",
    "family_history",
    "cancer_stage",
    "treatment_type",
    "smoking_status",
    "hypertension",
    "asthma",
    "cirrhosis",
    "other_cancer",
]

numerical_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Model pipeline
# 3. Advanced Ensemble Model
print(f"\n=== Training Advanced Ensemble Model ===")

# Create individual models
print(f"\n🤖 Training Individual Models...")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=8, random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000
    ),
}

# Voting ensemble
print(f"\nTraining Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()], voting="soft"
)
ensemble.fit(X_train_scaled, y_train)

# Make predictions
y_pred_ensemble = ensemble.predict(X_train_scaled)
ensemble_accuracy = accuracy_score(y_train, y_pred_ensemble)

# 4. Model Comparison
print(f"\n=== Model Comparison ===")
print(f"Advanced Ensemble: {ensemble_accuracy*100:.1f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))
print("Classification Report:")
print(classification_report(y_test, y_pred_ensemble))
