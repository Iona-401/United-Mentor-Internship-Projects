import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

FILE_PATH = "liver_cirrhosis_stage\liver_cirrhosis.csv"

# Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage

data = pd.read_csv(FILE_PATH)
print("Data Loaded Successfully")

missing_values = data.isnull().sum()
print(f"Number of missing values in each column: {missing_values}")

data.dropna(inplace=True)
print("Dropped Missing Values")

# Map data to numerical values
data["Status"] = data["Status"].map({"C": 0, "CL": 1, "D": 2})
data["Drug"] = data["Drug"].map({"D-penicillamine": 1, "Placebo": 0})
data["Sex"] = data["Sex"].map({"M": 1, "F": 0})
data["Ascites"] = data["Ascites"].map({"Y": 1, "N": 0})
data["Hepatomegaly"] = data["Hepatomegaly"].map({"Y": 1, "N": 0})
data["Spiders"] = data["Spiders"].map({"Y": 1, "N": 0})
data["Edema"] = data["Edema"].map({"Y": 2, "S": 1, "N": 0})
data["Stage"] = data["Stage"].map({1: 0, 2: 1, 3: 2})

X = data.drop("Stage", axis=1)
y = data["Stage"]

numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown = "ignore"))
])

# Define the preprocessing steps
numerical_features = ["N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]
categorical_features = ["Status", "Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create the model pipeline
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data Split into Training and Testing Sets")

# Train the model
model.fit(X_train, y_train)
print("Model Training Completed")

# Parameter tuning for Random Forest
#print("Starting Parameter Tuning...")
#param_grid = {
#    'classifier__max_depth': [6, 10, 15, None], 
#    'classifier__n_estimators': [200, 300, 500], 
#    'classifier__min_samples_split': [2, 5, 10],
#    'classifier__min_samples_leaf': [1, 2, 4],
#    'classifier__max_features': ['sqrt', 'log2', None]
#}
#grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
#grid_search.fit(X_train, y_train)
#print("Best Params:", grid_search.best_params_)
#print("Best Cross-validation Score:", grid_search.best_score_)

# Use the best model for predictions
#best_model = grid_search.best_estimator_

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\n=== Random Forest Model Evaluation (After Tuning) ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")
preprocessor = model.named_steps['preprocessor']
cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_features = numerical_features + list(cat_features)

rf_model = model.named_steps['classifier']
importances = rf_model.feature_importances_

# Display top 10 most important features
feature_importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance_df.head(10))

# Optional: Uncomment to plot feature importance
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 8))
# top_features = feature_importance_df.head(15)
# plt.barh(top_features['feature'], top_features['importance'])
# plt.xlabel("Feature Importance")
# plt.title("Random Forest Feature Importances (Top 15)")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()

# Saving the best model
joblib.dump(model, "random_forest_liver_cirrhosis_model.pkl")
print("Tuned model saved as random_forest_liver_cirrhosis_model.pkl")