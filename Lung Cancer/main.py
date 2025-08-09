import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
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
target_dist = df_clean['survived'].value_counts()
target_pct = df_clean['survived'].value_counts(normalize=True) * 100
print(f"Not Survived (0): {target_dist[0]} samples ({target_pct[0]:.1f}%)")
print(f"Survived (1): {target_dist[1]} samples ({target_pct[1]:.1f}%)")
print(f"Class imbalance ratio: {target_pct[0]/target_pct[1]:.2f}:1")

categorical_features = df_clean.select_dtypes(include=['object']).columns
numerical_features = df_clean.select_dtypes(include=['int64', 'float64']).columns

#print("Categorical column values:")
#for col in categorical_columns:
#    print(f"\n{col}:")
#    value_counts = df_clean[col].value_counts()
#    for value, count in value_counts.items():
#        print(f"  - {value}: {count} ({count/len(df_clean)*100:.1f}%)")

#Class Mapping for Easier UI Allocation
df_clean["gender"] = df_clean["gender"].map({
    "Male": 1,
    "Female": 0
})

df_clean["family_history"] = df_clean["family_history"].map({
    "No": 0,
    "Yes": 1
})

df_clean["cancer_stage"] = df_clean["cancer_stage"].map({
    "Stage I": 0,
    "Stage II": 1,
    "Stage III": 2,
    "Stage IV": 3
})

df_clean["treatment_type"] = df_clean["treatment_type"].map({
    "Surgery": 0,
    "Chemotherapy": 1,
    "Radiation": 2,
    "Combined": 3
})

df_clean["smoking_status"] = df_clean["smoking_status"].map({
    "Never Smoked": 0,
    "Passive Smoker": 1,
    "Former Smoker": 2,
    "Current Smoker": 3
})

# Prepare features and target
X = df_clean.drop("survived", axis=1)
y = df_clean["survived"]

print(f"\nFeature columns: {list(X.columns)}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
print(f"\nData Split:")
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# Pipeline setup
numerical_transformer = Pipeline([
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Model pipeline
print(f"\n=== Training Baseline Random Forest ===")
rf_baseline = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
    )
rf_baseline.fit(X_train, y_train)
y_pred_rf = rf_baseline.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Baseline RF Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.1f}%)")
    


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state = 42,
        n_jobs = -1,
        max_depth = None,
        max_features = "sqrt",
        min_samples_leaf = 1,
        min_samples_split = 2,
        n_estimators = 500
        
    ))
])