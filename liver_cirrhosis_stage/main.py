import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


FILE_PATH = "liver_cirrhosis_stage/liver_cirrhosis.csv"

# Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage

data = pd.read_csv(FILE_PATH)
print("Data Loaded Successfully")

missing_stage = data["Stage"].isnull().sum()
print(f"Number of missing values in 'Stage' column: {missing_stage}")

data.dropna(inplace=True)
print("Dropped Missing Values")

# Map data to numerical values
data["Status"] = data["Status"].map({"C": 0, "CL": 1, "D": 2})
data["Drug"] = data["Drug"].map({"D-penicillamine": 1, "Placebo": 0})
data["Sex"] = data["Sex"].map({"M": 1, "F": 0})
data["Ascites"] = data["Ascites"].map({"Y": 1, "N": 0})
data["Hepatomegaly"] = data["Hepatomegaly"].map({"Y": 1, "N": 0})
data["Spiders"] = data["Spiders"].map({"Y": 1, "N": 0})
data["Edema"] = data["Edema"].map({"Y": 1, "N": 0})
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
    ('classifier', XGBClassifier(
        eval_metric='mlogloss', 
        tree_method = "hist",
        device = "cuda",
        classifier__colsample_bytree = 0.8, 
        classifier__learning_rate = 0.1, 
        classifier__max_depth = 6, 
        classifier__n_estimators = 300, 
        classifier__subsample = 0.8
    ))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data Split into Training and Testing Sets")

# Train the model
model.fit(X_train, y_train)
print("Model Training Completed")

# Parameter tuning (optional)
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

# Saving the model
joblib.dump(model, "liver_cirrhosis_stage/xgboost_liver_cirrhosis_model.pkl")