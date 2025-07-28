import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

FILE_PATH = "thyroid_cancer/dataset.csv"

feature_names = ["Age", "Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy", "Thyroid Function", "Physical Examination", "Adenopathy", "Pathology", "Focality", "Risk", "T", "N", "M", "Stage", "Response", "Recurred"]

data = pd.read_csv(FILE_PATH)
print("Data Loaded Successfully")

missing_values = data.isnull().sum()
print(f"Number of missing values in each column: {missing_values}")

data.dropna(inplace=True)
print("Dropped Missing Values")

# Map data to numerical values
data["Gender"] = data["Gender"].map({
    "M": 1, 
    "F": 0
})
data["Smoking"] = data["Smoking"].map({
    "Yes": 1,
    "No": 0
})
data["Hx Smoking"] = data["Hx Smoking"].map({
    "Yes": 1, 
    "No": 0
})
data["Hx Radiothreapy"] = data["Hx Radiothreapy"].map({
    "Yes": 1, 
    "No": 0
})
data["Thyroid Function"] = data["Thyroid Function"].map({
    "Euthyroid": 4, 
    "Clinical Hyperthyroidism": 3,
    "Clinical Hypothyroidism": 2,
    "Subclinical Hyperthyroidism": 1, 
    "Subclinical Hypothyroidism": 0
})
data["Physical Examination"] = data["Physical Examination"].map({
    "Multinodular goiter": 4,
    "Single modular goiter-left": 3,
    "Single modular goiter-right": 2,
    "Diffuse goiter": 1,
    "Normal": 0
})
data["Adenopathy"] = data["Adenopathy"].map({
    "Extensive": 5,
    "Bilateral": 4,
    "Posterior": 3,
    "Left": 2,
    "Right": 1,
    "No": 0
})
data["Pathology"] = data["Pathology"].map({
    "Papillary": 3,
    "Follicular": 2,
    "Medullary": 1,
    "Hurthle Cell": 0
})
data["Focality"] = data["Focality"].map({
    "Unifocal": 1,
    "Multifocal": 0
})
data["Risk"] = data["Risk"].map({
    "High": 2,
    "Intermediate": 1,
    "Low": 0
})
data["T"] = data["T"].map({
    "T4b": 6,
    "T4a": 5,
    "T3b": 4,
    "T3a": 3,
    "T2": 2,
    "T1b": 1,
    "T1a": 0
})
data["N"] = data["N"].map({
    "N1b": 2,
    "N1a": 1,
    "N0": 0
})
data["M"] = data["M"].map({
    "M1": 1,
    "M0": 0
})
data["Stage"] = data["Stage"].map({
    "IVB": 4,
    "IVA": 3,
    "III": 2,
    "II": 1,
    "I": 0
})
data["Response"] = data["Response"].map({
    "Excellent": 3,
    "Biochemical Incomplete": 2,
    "Structural Incomplete": 1,
    "Intermediate": 0
})
data["Recurred"] = data["Recurred"].map({
    "Yes": 1,
    "No": 0
})


X = data.drop("Recurred", axis=1)
y = data["Recurred"]

numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown = "ignore"))
])

# Define the preprocessing steps
numerical_features = ["Age"]

categorical_features = [ "Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy", "Thyroid Function", "Physical Examination", "Adenopathy", "Pathology", "Focality", "Risk", "T", "N", "M", "Stage", "Response"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        eval_metric='mlogloss', 
        tree_method = "hist",
        device = "cuda",
        classifier__colsample_bytree = 0.8, 
        classifier__learning_rate = 0.05, 
        classifier__max_depth = 4, 
        classifier__n_estimators = 100, 
        classifier__subsample = 1.0
    ))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data Split into Training and Testing Sets")

# Train the model
model.fit(X_train, y_train)
print("Model Training Completed")

# Parameter tuning
#param_grid = {'classifier__max_depth': [3, 4, 5, 6], 'classifier__learning_rate': [0.01, 0.05, 0.1], 'classifier__n_estimators': [100, 200, 300], 'classifier__subsample': [0.8, 1.0], 'classifier__colsample_bytree': [0.8, 1.0]}
#grid_search = GridSearchCV(model, param_grid, cv = 3, scoring = "accuracy", verbose=1, n_jobs=-1)
#grid_search.fit(X_train, y_train)
#print("Best Params:", grid_search.best_params_)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("=== XGBoost Model Evaluation ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Plot
#preprocessor = model.named_steps['preprocessor']
#cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
#all_features = numerical_features + list(cat_features)

#xgb = model.named_steps['classifier']
#importances = xgb.feature_importances_

# Plot
#plt.figure(figsize=(10,6))
#plt.barh(all_features, importances)
#plt.xlabel("Feature Importance")
#plt.title("XGBoost Feature Importances")
#plt.show()

# Saving the model
joblib.dump(model, "thyroid_cancer/xgboost_thyroid_cancer_model.pkl")