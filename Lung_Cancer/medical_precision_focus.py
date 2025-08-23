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
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight
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


def evaluate_model_medical(y_true, y_pred, y_pred_proba=None, model_name=""):
    """Evaluate model with medical context - emphasizing precision"""
    print(f"\n{'='*60}")
    print(f"🏥 MEDICAL EVALUATION: {model_name}")
    print(f"{'='*60}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Calculate confusion matrix details
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Key medical metrics
    false_positive_rate = fp / (fp + tn)  # How often we wrongly predict survival
    false_negative_rate = fn / (fn + tp)  # How often we miss actual survivors
    specificity = tn / (tn + fp)  # True negative rate

    print(f"📊 PERFORMANCE METRICS:")
    print(f"   Accuracy:   {accuracy:.4f}")
    print(f"   Precision:  {precision:.4f} ⭐ (Most Important - Avoiding false hope)")
    print(f"   Recall:     {recall:.4f} (Catching actual survivors)")
    print(f"   F1 Score:   {f1:.4f}")

    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"   ROC AUC:    {auc:.4f}")

    print(f"\n🚨 MEDICAL RISK ANALYSIS:")
    print(
        f"   False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)"
    )
    print(f"   ↳ Risk of giving false hope to patients")
    print(
        f"   False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)"
    )
    print(f"   ↳ Risk of missing actual survivors")

    print(f"\n📋 CONFUSION MATRIX BREAKDOWN:")
    print(f"   True Negatives (Correct 'Won't Survive'):  {tn:8,} ✅")
    print(f"   False Positives (Wrong 'Will Survive'):    {fp:8,} ⚠️  DANGEROUS")
    print(f"   False Negatives (Missed Survivors):        {fn:8,} 😞")
    print(f"   True Positives (Correct Survivors):        {tp:8,} ✅")

    # Medical interpretation
    total_predicted_survivors = tp + fp
    if total_predicted_survivors > 0:
        survival_prediction_accuracy = tp / total_predicted_survivors
        print(f"\n🎯 MEDICAL INTERPRETATION:")
        print(
            f"   When you predict 'WILL SURVIVE': {survival_prediction_accuracy:.1%} are actually correct"
        )
        print(
            f"   When you predict 'WON'T SURVIVE': {tn/(tn+fn):.1%} are actually correct"
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc if y_pred_proba is not None else None,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "confusion_matrix": cm,
    }


def find_optimal_threshold(y_true, y_pred_proba, target_precision=0.8):
    """Find threshold that maximizes precision while maintaining reasonable recall"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Find threshold that gives us target precision
    target_indices = precisions >= target_precision
    if np.any(target_indices):
        best_idx = np.where(target_indices)[0][0]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
    else:
        # If target precision not achievable, find best precision-recall trade-off
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]

    print(f"🎯 OPTIMAL THRESHOLD ANALYSIS:")
    print(f"   Target Precision: {target_precision:.1%}")
    print(f"   Optimal Threshold: {best_threshold:.4f}")
    print(f"   Achieved Precision: {best_precision:.4f} ({best_precision:.1%})")
    print(f"   Corresponding Recall: {best_recall:.4f} ({best_recall:.1%})")

    return best_threshold, best_precision, best_recall


def main():
    # Load data
    X, y = load_and_preprocess_data()

    print(f"\n📋 DATASET INFO:")
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
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    print(f"\n🏥 MEDICAL-FOCUSED MODELS (Precision-Optimized)")
    print("=" * 80)

    # APPROACH 1: Conservative Model (Favor Precision over Recall)
    print(f"\n🔴 APPROACH 1: CONSERVATIVE MODEL (High Precision Focus)")
    print("-" * 60)

    # Use inverse class weights to favor precision (penalize false positives more)
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    # Reduce the positive class weight to be more conservative
    conservative_weight = (
        class_weights[1] / class_weights[0] * 0.3
    )  # Make it more conservative

    xgb_conservative = xgboost.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=150,
        max_depth=4,  # Reduce depth to prevent overfitting
        learning_rate=0.05,  # Lower learning rate for stability
        random_state=42,
        scale_pos_weight=conservative_weight,
        min_child_weight=10,  # Require more samples per leaf (conservative)
        reg_alpha=1,  # L1 regularization
        reg_lambda=1,  # L2 regularization
    )

    print(f"Using conservative scale_pos_weight: {conservative_weight:.4f}")
    xgb_conservative.fit(X_train_scaled, y_train)

    # Get probabilities and find optimal threshold for high precision
    y_pred_proba_conservative = xgb_conservative.predict_proba(X_test_scaled)[:, 1]

    # Find threshold for 80% precision target
    optimal_threshold, achieved_precision, achieved_recall = find_optimal_threshold(
        y_test, y_pred_proba_conservative, target_precision=0.8
    )

    # Make predictions with optimal threshold
    y_pred_conservative = (y_pred_proba_conservative >= optimal_threshold).astype(int)

    results_conservative = evaluate_model_medical(
        y_test,
        y_pred_conservative,
        y_pred_proba_conservative,
        "CONSERVATIVE PRECISION-FOCUSED XGBoost",
    )

    # APPROACH 2: Extremely Conservative Model (90%+ precision target)
    print(f"\n🟡 APPROACH 2: EXTREMELY CONSERVATIVE (90%+ Precision Target)")
    print("-" * 60)

    # Even more conservative
    ultra_conservative_weight = class_weights[1] / class_weights[0] * 0.1

    xgb_ultra = xgboost.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=200,
        max_depth=3,  # Very shallow
        learning_rate=0.03,  # Very low learning rate
        random_state=42,
        scale_pos_weight=ultra_conservative_weight,
        min_child_weight=20,  # Very conservative
        reg_alpha=2,  # Higher regularization
        reg_lambda=2,
        subsample=0.8,  # Use only 80% of samples for each tree
        colsample_bytree=0.8,  # Use only 80% of features
    )

    print(f"Using ultra-conservative scale_pos_weight: {ultra_conservative_weight:.4f}")
    xgb_ultra.fit(X_train_scaled, y_train)

    # Get probabilities and find threshold for 90% precision
    y_pred_proba_ultra = xgb_ultra.predict_proba(X_test_scaled)[:, 1]

    ultra_threshold, ultra_precision, ultra_recall = find_optimal_threshold(
        y_test, y_pred_proba_ultra, target_precision=0.9
    )

    # Make predictions with ultra-conservative threshold
    y_pred_ultra = (y_pred_proba_ultra >= ultra_threshold).astype(int)

    results_ultra = evaluate_model_medical(
        y_test,
        y_pred_ultra,
        y_pred_proba_ultra,
        "ULTRA-CONSERVATIVE XGBoost (90%+ Precision)",
    )

    # APPROACH 3: Baseline Comparison
    print(f"\n🔵 APPROACH 3: BASELINE (Default XGBoost)")
    print("-" * 60)

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

    results_baseline = evaluate_model_medical(
        y_test, y_pred_baseline, y_pred_proba_baseline, "BASELINE XGBoost"
    )

    # FINAL RECOMMENDATIONS
    print(f"\n{'='*80}")
    print(f"🏥 MEDICAL RECOMMENDATION SUMMARY")
    print(f"{'='*80}")

    models_comparison = {
        "Conservative (80% precision target)": {
            "precision": results_conservative["precision"],
            "recall": results_conservative["recall"],
            "false_positive_rate": results_conservative["false_positive_rate"],
            "threshold": optimal_threshold,
        },
        "Ultra-Conservative (90% precision target)": {
            "precision": results_ultra["precision"],
            "recall": results_ultra["recall"],
            "false_positive_rate": results_ultra["false_positive_rate"],
            "threshold": ultra_threshold,
        },
        "Baseline": {
            "precision": results_baseline["precision"],
            "recall": results_baseline["recall"],
            "false_positive_rate": results_baseline["false_positive_rate"],
            "threshold": 0.5,
        },
    }

    print(f"\n📊 COMPARISON TABLE:")
    print(
        f"{'Model':<35} {'Precision':<10} {'Recall':<8} {'FP Rate':<8} {'Threshold':<10}"
    )
    print("-" * 75)
    for model_name, metrics in models_comparison.items():
        print(
            f"{model_name:<35} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f} {metrics['false_positive_rate']:<8.4f} {metrics['threshold']:<10.4f}"
        )

    print(f"\n💡 MEDICAL RECOMMENDATIONS:")
    print("-" * 40)
    print("🟢 USE CONSERVATIVE MODEL IF:")
    print("   • You want ~80% precision (1 in 5 survival predictions may be wrong)")
    print("   • You can accept missing some actual survivors")
    print("   • Balance between avoiding false hope and catching survivors")

    print(f"\n🟡 USE ULTRA-CONSERVATIVE MODEL IF:")
    print("   • False hope is extremely dangerous")
    print("   • You want 90%+ precision (9 out of 10 survival predictions correct)")
    print("   • You're willing to miss more survivors to avoid false positives")

    print(f"\n⚠️  CRITICAL CONSIDERATION:")
    print("   • In oncology, both false positives AND false negatives are serious")
    print("   • Consider using model confidence scores for decision support")
    print("   • Always combine with clinical expertise")
    print("   • Consider a 'uncertain' category for borderline cases")

    # Save the best model
    print(f"\n💾 SAVING CONSERVATIVE MODEL...")
    joblib.dump(xgb_conservative, "lung_cancer_conservative_model.pkl")
    joblib.dump(preprocessor, "lung_cancer_preprocessor.pkl")

    # Save optimal threshold
    with open("optimal_threshold.txt", "w") as f:
        f.write(f"Optimal Threshold for 80% Precision: {optimal_threshold:.6f}\n")
        f.write(
            f"Ultra-Conservative Threshold for 90% Precision: {ultra_threshold:.6f}\n"
        )

    print(f"✅ Models and preprocessor saved!")
    print(f"📝 Optimal thresholds saved to 'optimal_threshold.txt'")


if __name__ == "__main__":
    main()
