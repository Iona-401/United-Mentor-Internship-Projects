import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

FILE_PATH = "thyroid_cancer\dataset.csv"

feature_names = [
    "Age",
    "Gender",
    "Smoking",
    "Hx Smoking",
    "Hx Radiothreapy",
    "Thyroid Function",
    "Physical Examination",
    "Adenopathy",
    "Pathology",
    "Focality",
    "Risk",
    "T",
    "N",
    "M",
    "Stage",
    "Response",
    "Recurred",
]

data = pd.read_csv(FILE_PATH)
print("Data Loaded Successfully")

missing_values = data.isnull().sum()
print(f"Number of missing values in each column: {missing_values}")

data.dropna(inplace=True)
print("Dropped Missing Values")

X = data.drop("Recurred", axis=1)
y = data["Recurred"]

numeric_transformer = Pipeline([("scaler", StandardScaler())])

categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Define the preprocessing steps
numerical_features = ["Age"]

categorical_features = [
    "Gender",
    "Smoking",
    "Hx Smoking",
    "Hx Radiothreapy",
    "Thyroid Function",
    "Physical Examination",
    "Adenopathy",
    "Pathology",
    "Focality",
    "Risk",
    "T",
    "N",
    "M",
    "Stage",
    "Response",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Create the model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                max_depth=5,
                max_features="sqrt",
                min_samples_leaf=1,
                min_samples_split=5,
                n_estimators=200,
                random_state=42,
            ),
        ),
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data Split into Training and Testing Sets")

# Train the model
model.fit(X_train, y_train)
print("Model Training Completed")

# Parameter tuning for Random Forest
# param_grid = {
#    'classifier__n_estimators': [100, 200, 300],
#    'classifier__max_depth': [3, 4, 5, 6, None],
#    'classifier__min_samples_split': [2, 5, 10],
#    'classifier__min_samples_leaf': [1, 2, 4],
#    'classifier__max_features': ['sqrt', 'log2']
# }

# print("Starting Grid Search for Random Forest...")
# grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# print("Best Params:", grid_search.best_params_)
# print("Best Cross-validation Score:", grid_search.best_score_)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("=== Random Forest Model Evaluation ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Plot
# preprocessor = model.named_steps['preprocessor']
# cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
# all_features = numerical_features + list(cat_features)

# xgb = model.named_steps['classifier']
# importances = xgb.feature_importances_

# Plot
# plt.figure(figsize=(10,6))
# plt.barh(all_features, importances)
# plt.xlabel("Feature Importance")
# plt.title("XGBoost Feature Importances")
# plt.show()

# Saving the model
joblib.dump(model, "random_forest_thyroid_cancer_model.pkl")
