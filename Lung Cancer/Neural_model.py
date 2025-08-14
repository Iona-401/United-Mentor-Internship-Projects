import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

FILE_PATH = "Lung Cancer/dataset_med.csv"
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


def create_improved_neural_network(input_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(256, activation="relu", kernel_initializer="he_normal"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )
    return model


numerical_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Prepare features and target
X = df_clean.drop("survived", axis=1)
y = df_clean["survived"]
print(f"\nFeature columns: {list(X.columns)}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

print(f"\nData split:")
print(f"Training set shape: {X_train_scaled.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test_scaled.shape}, {y_test.shape}")

print(f"\nCalculating class weights for imbalanced data...")
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

print(f"\nCreating improved neural network...")
model = create_improved_neural_network(X_train_scaled.shape[1])
model.summary()

# FIXED: Better training with callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
    ),
]

# CRITICAL FIX: Train with class weights and proper validation
print(f"\nTraining neural network with class balancing...")
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,  # More epochs
    batch_size=1024,  # Better batch size
    validation_split=0.2,  # Use validation split, not test data
    class_weight=class_weight_dict,  # CRITICAL: Handle imbalanced data
    callbacks=callbacks,
    verbose=1,
)

# PROPER EVALUATION on test set
print(f"\nEvaluating on test set...")
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype("int32")

# Calculate comprehensive metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== NEURAL NETWORK RESULTS ===")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"[[{cm[0][0]:,}, {cm[0][1]:,}]")
print(f" [{cm[1][0]:,}, {cm[1][1]:,}]]")

# Check if model is actually predicting survival cases
survived_predictions = sum(y_pred.flatten())
actual_survivors = sum(y_test)
print(f"\nPrediction Analysis:")
print(
    f"Predicted Survivals: {survived_predictions:,} ({survived_predictions/len(y_test)*100:.1f}%)"
)
print(
    f"Actual Survivals: {actual_survivors:,} ({actual_survivors/len(y_test)*100:.1f}%)"
)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# THRESHOLD OPTIMIZATION
print(f"\nOptimizing prediction threshold...")
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    f1_thresh = f1_score(y_test, y_pred_thresh)

    if f1_thresh > best_f1:
        best_f1 = f1_thresh
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f} with F1-score: {best_f1:.3f}")
