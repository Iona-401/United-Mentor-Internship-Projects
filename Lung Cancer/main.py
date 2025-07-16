
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

file_path = "Lung Cancer/dataset_med.csv"

df = pd.read_csv(file_path)
df = df.drop(columns=["id", "country", "diagnosis_date", "end_treatment_date"])

df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    
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

X = df.drop(columns=["survived"])
y = df["survived"]

numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_features = ["age",
    "bmi", 
    "cholesterol_level", 
    "hypertension", 
    "asthma", 
    "cirrhosis", 
    "other_cancer"
]
categorical_features = [
    "gender", 
    #"country",
    "cancer_stage", 
    "family_history",
    "smoking_status", 
    "treatment_type"
]

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="passthrough")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model =  Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1))
])
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(f"\n=== Random Forest Model ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(rf_model, "Lung Cancer/lung_cancer_rf_model.pkl")