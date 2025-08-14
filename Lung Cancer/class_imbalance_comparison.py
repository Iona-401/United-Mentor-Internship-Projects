import pandas as pd
import numpy as np
import joblib
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
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
import time
import warnings

warnings.filterwarnings("ignore")

FILE_PATH = "Lung Cancer/dataset_med.csv"


def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("Loading and preprocessing data...")
    df = pd.read_csv(FILE_PATH)
    print(f"Original Dataset Shape: {df.shape}")

    # Drop unnecessary columns
    columns_to_drop = ["id", "country", "diagnosis_date", "end_treatment_date"]
    df_clean = df.drop(columns=columns_to_drop)

    # Handle missing values
    df_clean.dropna(inplace=True)

    # Class Mapping for Easier UI Allocation
    df_clean["hypertension"] = df_clean["hypertension"].map({0: "No", 1: "Yes"})
    df_clean["asthma"] = df_clean["asthma"].map({0: "No", 1: "Yes"})
    df_clean["cirrhosis"] = df_clean["cirrhosis"].map({0: "No", 1: "Yes"})
    df_clean["other_cancer"] = df_clean["other_cancer"].map({0: "No", 1: "Yes"})

    # Prepare features and target
    X = df_clean.drop("survived", axis=1)
    y = df_clean["survived"]

    return X, y


def create_preprocessor():
    """Create preprocessing pipeline"""
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

    numerical_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name=""):
    """Evaluate model performance"""
    print(f"\n{'='*50}")
    print(f"Results for: {model_name}")
    print(f"{'='*50}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(
        f"Precision: {precision:.4f} (TP/(TP+FP)) - Of predicted survivors, how many actually survived"
    )
    print(
        f"Recall:    {recall:.4f} (TP/(TP+FN)) - Of actual survivors, how many were correctly predicted"
    )
    print(f"F1 Score:  {f1:.4f} (Harmonic mean of precision and recall)")

    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"ROC AUC:   {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Not Survived  Survived")
    print(f"Actual Not S.    {cm[0,0]:8d}   {cm[0,1]:8d}")
    print(f"Actual Surv.     {cm[1,0]:8d}   {cm[1,1]:8d}")

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print(f"\nSpecificity: {specificity:.4f} (True Negative Rate)")
    print(f"Sensitivity: {recall:.4f} (Same as Recall)")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": (
            roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
        ),
        "confusion_matrix": cm,
    }


def main():
    # Load data
    X, y = load_and_preprocess_data()

    print(f"\nDataset Info:")
    print(f"Total samples: {len(X)}")
    target_dist = y.value_counts()
    target_pct = y.value_counts(normalize=True) * 100
    print(f"Not Survived (0): {target_dist[0]} samples ({target_pct[0]:.1f}%)")
    print(f"Survived (1): {target_dist[1]} samples ({target_pct[1]:.1f}%)")
    print(f"Class imbalance ratio: {target_pct[0]/target_pct[1]:.2f}:1")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create preprocessor
    preprocessor = create_preprocessor()

    # Store results
    results = {}

    # 1. BASELINE MODEL (No class imbalance handling)
    print(f"\n🔵 APPROACH 1: BASELINE (No imbalance handling)")
    print("-" * 60)

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    start_time = time.time()
    xgb_baseline = xgboost.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    xgb_baseline.fit(X_train_scaled, y_train)

    y_pred_baseline = xgb_baseline.predict(X_test_scaled)
    y_pred_proba_baseline = xgb_baseline.predict_proba(X_test_scaled)[:, 1]

    results["baseline"] = evaluate_model(
        y_test, y_pred_baseline, y_pred_proba_baseline, "BASELINE XGBoost"
    )
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    # 2. CLASS WEIGHT BALANCING (Bias the model)
    print(f"\n🟡 APPROACH 2: CLASS WEIGHT BALANCING (Bias Model)")
    print("-" * 60)

    start_time = time.time()
    # Calculate class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Calculated class weights: {weight_dict}")

    xgb_weighted = xgboost.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=class_weights[1]
        / class_weights[0],  # XGBoost way of handling class weights
    )
    xgb_weighted.fit(X_train_scaled, y_train)

    y_pred_weighted = xgb_weighted.predict(X_test_scaled)
    y_pred_proba_weighted = xgb_weighted.predict_proba(X_test_scaled)[:, 1]

    results["weighted"] = evaluate_model(
        y_test, y_pred_weighted, y_pred_proba_weighted, "CLASS WEIGHTED XGBoost"
    )
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    # 3. RANDOM UNDERSAMPLING (Reduce majority class)
    print(f"\n🟠 APPROACH 3: RANDOM UNDERSAMPLING (Reduce Majority Class)")
    print("-" * 60)

    start_time = time.time()
    undersampler = RandomUnderSampler(random_state=42)
    X_train_under, y_train_under = undersampler.fit_resample(X_train_scaled, y_train)

    print(f"Original training set: {X_train_scaled.shape[0]} samples")
    print(f"After undersampling: {X_train_under.shape[0]} samples")
    print(f"Class distribution after undersampling: {np.bincount(y_train_under)}")

    xgb_under = xgboost.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    xgb_under.fit(X_train_under, y_train_under)

    y_pred_under = xgb_under.predict(X_test_scaled)
    y_pred_proba_under = xgb_under.predict_proba(X_test_scaled)[:, 1]

    results["undersampling"] = evaluate_model(
        y_test, y_pred_under, y_pred_proba_under, "UNDERSAMPLED XGBoost"
    )
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    # 4. SMOTE OVERSAMPLING (Synthetic minority oversampling)
    print(f"\n🟢 APPROACH 4: SMOTE OVERSAMPLING (Synthetic Minority Samples)")
    print("-" * 60)

    start_time = time.time()
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"Original training set: {X_train_scaled.shape[0]} samples")
    print(f"After SMOTE: {X_train_smote.shape[0]} samples")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_smote)}")

    xgb_smote = xgboost.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    xgb_smote.fit(X_train_smote, y_train_smote)

    y_pred_smote = xgb_smote.predict(X_test_scaled)
    y_pred_proba_smote = xgb_smote.predict_proba(X_test_scaled)[:, 1]

    results["smote"] = evaluate_model(
        y_test, y_pred_smote, y_pred_proba_smote, "SMOTE XGBoost"
    )
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    # 5. BALANCED RANDOM FOREST (Built-in balancing)
    print(f"\n🟣 APPROACH 5: BALANCED RANDOM FOREST (Built-in Balancing)")
    print("-" * 60)

    start_time = time.time()
    brf = BalancedRandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    brf.fit(X_train_scaled, y_train)

    y_pred_brf = brf.predict(X_test_scaled)
    y_pred_proba_brf = brf.predict_proba(X_test_scaled)[:, 1]

    results["balanced_rf"] = evaluate_model(
        y_test, y_pred_brf, y_pred_proba_brf, "BALANCED RANDOM FOREST"
    )
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    # FINAL COMPARISON
    print(f"\n{'='*80}")
    print(f"📊 FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")

    comparison_df = pd.DataFrame(
        {
            "Method": [
                "Baseline",
                "Class Weighted",
                "Undersampling",
                "SMOTE",
                "Balanced RF",
            ],
            "Accuracy": [
                results["baseline"]["accuracy"],
                results["weighted"]["accuracy"],
                results["undersampling"]["accuracy"],
                results["smote"]["accuracy"],
                results["balanced_rf"]["accuracy"],
            ],
            "Precision": [
                results["baseline"]["precision"],
                results["weighted"]["precision"],
                results["undersampling"]["precision"],
                results["smote"]["precision"],
                results["balanced_rf"]["precision"],
            ],
            "Recall": [
                results["baseline"]["recall"],
                results["weighted"]["recall"],
                results["undersampling"]["recall"],
                results["smote"]["recall"],
                results["balanced_rf"]["recall"],
            ],
            "F1": [
                results["baseline"]["f1"],
                results["weighted"]["f1"],
                results["undersampling"]["f1"],
                results["smote"]["f1"],
                results["balanced_rf"]["f1"],
            ],
            "AUC": [
                results["baseline"]["auc"],
                results["weighted"]["auc"],
                results["undersampling"]["auc"],
                results["smote"]["auc"],
                results["balanced_rf"]["auc"],
            ],
        }
    )

    print(comparison_df.round(4))

    print(f"\n🎯 RECOMMENDATIONS:")
    print("-" * 40)

    best_f1_idx = comparison_df["F1"].idxmax()
    best_auc_idx = comparison_df["AUC"].idxmax()
    best_recall_idx = comparison_df["Recall"].idxmax()

    print(
        f"• Best F1 Score: {comparison_df.loc[best_f1_idx, 'Method']} ({comparison_df.loc[best_f1_idx, 'F1']:.4f})"
    )
    print(
        f"• Best AUC Score: {comparison_df.loc[best_auc_idx, 'Method']} ({comparison_df.loc[best_auc_idx, 'AUC']:.4f})"
    )
    print(
        f"• Best Recall (Sensitivity): {comparison_df.loc[best_recall_idx, 'Method']} ({comparison_df.loc[best_recall_idx, 'Recall']:.4f})"
    )

    print(f"\n📝 ANALYSIS:")
    print(
        "- High Precision: Good at avoiding false positives (predicting survival when patient won't)"
    )
    print(
        "- High Recall: Good at catching true positives (correctly predicting actual survivors)"
    )
    print("- High F1: Good balance between precision and recall")
    print("- High AUC: Good overall discriminative ability")

    print(f"\n💡 For medical applications, you typically want:")
    print(
        "- High RECALL if missing survivors is costly (better to overpredict survival)"
    )
    print("- High PRECISION if false hope is problematic (better to be conservative)")
    print("- Balanced F1 for overall good performance")


if __name__ == "__main__":
    main()
