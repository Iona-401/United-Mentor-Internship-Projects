
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def load_data(file_path = "Lung Cancer/dataset_med.csv"):
    """Load the dataset from a CSV file.

    Args:
        file_path (str, optional): Path to the CSV file. Defaults to "Lung Cancer/dataset_med.csv".

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path)

def clean_data(data):
    """Clean the dataset by handling missing values and encoding categorical variables.

    Args:
        data (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    df = data.drop(columns=["id", "country", "diagnosis_date", "end_treatment_date"])
    #df = df.drop(columns=["bmi", "cholesterol_level", "asthma", "hypertension", "cirrhosis", "other_cancer"])
    return df

def map_categorical_columns(df):
    df = df.copy()
    
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

    return df

def split_data(data, target = "survived"):
    """Split the dataset into features and target variable.

    Args:
        data (pd.DataFrame): The input dataset.
        target (str, optional): The target variable name. Defaults to "survived".

    Returns:
        tuple: A tuple containing the features (X) and target (y).
    """
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

def get_preprocessor(numerical_features, categorical_features):
    """Create a preprocessing pipeline for numerical and categorical features.

    Args:
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        ColumnTransformer: The preprocessing pipeline.
    """
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="passthrough")

    return preprocessor

def evaluate_model(name, model, X_test, y_test):
    """Evaluate the model's performance.

    Args:
        name (str): The name of the model.
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target.
    """
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def build_random_forest_pipeline(preprocessor):
    """Build a Random Forest pipeline.

    Args:
        preprocessor (ColumnTransformer): The preprocessing pipeline.

    Returns:
        Pipeline: The Random Forest pipeline.
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1))
    ])

def main():
    """Main function to execute the data loading, cleaning, preprocessing, model training, and evaluation.
    """
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)
    df = map_categorical_columns(df)

    print("Splitting features and target...")
    X, y = split_data(df)

    print("Feature columns:", X.columns.tolist())
    print("Target column:", y.name)
    print("Data loaded and cleaned successfully.")
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    print("Preparing preprocessing pipeline...")
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
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    print("Preprocessor created.")

    print("Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    print("Training Random Forest...")
    rf_model =  build_random_forest_pipeline(preprocessor)
    rf_model.fit(X_train, y_train)
    print("Random Forest trained.")

    print("Evaluating models...")
    evaluate_model("Random Forest", rf_model, X_test, y_test)

<<<<<<< HEAD
<<<<<<< HEAD
    # Save the trained model
    scaler = StandardScaler()
    joblib.dump(rf_model, "Lung Cancer/lung_cancer_rf_model.pkl")
=======
    plot_logistic_coefficients(log_model, X.columns.tolist())
    # Save the trained model
    joblib.dump(log_model, "lung_cancer_model.pkl")
>>>>>>> parent of 4409619 (Created Input Page for Lung Caner Model)
=======
    plot_logistic_coefficients(log_model, X.columns.tolist())
    # Save the trained model
    joblib.dump(log_model, "lung_cancer_model.pkl")
>>>>>>> parent of 4409619 (Created Input Page for Lung Caner Model)

if __name__ == "__main__":
    main()