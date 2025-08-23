import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import BalancedRandomForestClassifier

import warnings

warnings.filterwarnings("ignore")

FILE_PATH = "Lung Cancer\dataset_med.csv"

df = pd.read_csv(FILE_PATH)
print(f"Original Dataset Shape: {df.shape}")
print("Data Loaded Successfully")

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

# Class Mapping for Easier UI Allocation
df_clean["hypertension"] = df_clean["hypertension"].map({0: "No", 1: "Yes"})
df_clean["asthma"] = df_clean["asthma"].map({0: "No", 1: "Yes"})
df_clean["cirrhosis"] = df_clean["cirrhosis"].map({0: "No", 1: "Yes"})
df_clean["other_cancer"] = df_clean["other_cancer"].map({0: "No", 1: "Yes"})

print(f"\nDataset Info:")
print(f"Columns: {list(df_clean.columns)}")
print(f"Data types:\n{df_clean.dtypes}")

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


xgb = xgboost.XGBClassifier(
    use_label_encoder=False,
    device="gpu",
    booster="gbtree",
    subsample=0.8,
    n_estimators=200,
    min_child_weight=1,
    max_depth=10,
    learning_rate=0.2,
    gamma=0.1,
    colsample_bytree=1.0,
)
xgb.fit(X_train_scaled, y_train)
# temp
# Evaluation
y_pred = xgb.predict(X_test_scaled)
print(f"Test set predictions: {y_pred}")
print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print(
    f"ROC AUC Score: {roc_auc_score(y_test, xgb.predict_proba(X_test_scaled)[:, 1]):.2f}"
)

# Feature importance analysis
importances = xgb.feature_importances_
num_features = ["age", "bmi", "cholesterol_level"]
cat_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
    categorical_features
)
feature_names = numerical_features + list(cat_names)

# Create a DataFrame for easy analysis
feat_imp_df = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values(by="importance", ascending=False)

print(f"\nFeature Importance Analysis (XGBoost):")
print("=" * 50)
print(f"Top 20 Most Important Features:")
for i, (_, row) in enumerate(feat_imp_df.head(20).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:30s}: {row['importance']:.4f}")

print(f"\nTop 5 Feature Summary:")
print("-" * 30)
for i, (_, row) in enumerate(feat_imp_df.head(5).iterrows(), 1):
    print(
        f"{i}. {row['feature']}: {row['importance']:.4f} ({row['importance']*100:.2f}%)"
    )

total_importance = feat_imp_df.head(10)["importance"].sum()
print(
    f"\nTop 10 features account for {total_importance:.4f} ({total_importance*100:.2f}%) of total importance"
)
