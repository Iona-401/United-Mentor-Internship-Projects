
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "Lung Cancer/dataset_med.csv"

df = pd.read_csv(file_path)
print(f"Dataset shape: {df.shape}")
print("Dataset Loaded Successfully")

df = df.drop(columns=["id", "country", "diagnosis_date", "end_treatment_date"])
print("Unnecessary columns dropped.")

df["gender"] = df["gender"].map({
    "Male": 0, 
    "Female": 1
})  
df["cancer_stage"] = df["cancer_stage"].map({
    "Stage I": 0,
    "Stage II": 1,
    "Stage III": 2,
    "Stage IV": 3
})  
df["smoking_status"] = df["smoking_status"].map({
    "Never Smoked": 0,
    "Former Smoker": 1,
    "Current Smoker": 2
})    
df["treatment_type"] = df["treatment_type"].map({
    "Surgery": 0,
    "Chemotherapy": 1,
    "Radiation": 2,
    "Immunotherapy": 3
})
df["family_history"] = df["family_history"].map({
    "Yes": 1,
    "No": 0
})
    
print("Categorical columns mapped to numerical values.")

X = df.drop(columns=["survived"])
y = df["survived"]

# Drop rows with missing values
X = X.dropna()
y = y.loc[X.index]

numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_features = ["age",
    "gender",
    "family_history",
    "bmi",
    "cholesterol_level",
    "hypertension",
    "asthma",
    "cirrhosis",
    "other_cancer"
]
categorical_features = [
    "cancer_stage", 
    "smoking_status", 
    "treatment_type"
]

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="passthrough")

neg, pos = (y == 0).sum(), (y == 1).sum()
scale_to_weight = neg / pos
print(f"Class distribution: {neg} negatives, {pos} positives. Scale to weight: {scale_to_weight:.2f}")
xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        use_label_encoder=False, 
        eval_metric = "logloss", 
        scale_pos_weight = scale_to_weight, 
        threshold = 0.52,
        max_depth=4,
        min_child_weight=5
    ))
])

# Split data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE only to the training data
# Apply SMOTE only if you want to use it for RandomForest
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# print("Applied SMOTE to the training data.")

# Train XGBoost on original data
xgb_model.fit(X_train, y_train)
print("Model trained successfully.")

# Make predictions
y_pred = xgb_model.predict(X_test)
print("Predictions made on the test set.")
y_proba = xgb_model.predict_proba(X_test)[:, 1]
thresh = 0.52
y_pred_thresh = (y_proba > thresh).astype(int)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("====XGBoost Model Evaluation====")
print(f"Model accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
# Plot or inspect values to pick a threshold with desired precision
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label='Precision', marker='o')
plt.plot(thresholds, recall[:-1], label='Recall', marker='o')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid()
plt.show()