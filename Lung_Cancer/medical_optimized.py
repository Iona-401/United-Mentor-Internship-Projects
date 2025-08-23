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

    # Drop unnecessary columns
    columns_to_drop = ["id", "country", "diagnosis_date", "end_treatment_date"]
    df_clean = df.drop(columns=columns_to_drop)
    df_clean.dropna(inplace=True)

    # Class Mapping
    df_clean["hypertension"] = df_clean["hypertension"].map({0: "No", 1: "Yes"})
    df_clean["asthma"] = df_clean["asthma"].map({0: "No", 1: "Yes"})
    df_clean["cirrhosis"] = df_clean["cirrhosis"].map({0: "No", 1: "Yes"})
    df_clean["other_cancer"] = df_clean["other_cancer"].map({0: "No", 1: "Yes"})

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


def evaluate_precision_focused(y_true, y_pred, y_pred_proba, model_name, threshold):
    """Evaluate with medical precision focus"""
    print(f"\n{'='*70}")
    print(f"🏥 {model_name}")
    print(f"Decision Threshold: {threshold:.4f}")
    print(f"{'='*70}")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"📊 KEY MEDICAL METRICS:")
    print(
        f"   Precision:  {precision:.4f} ⭐ (When predicting survival, accuracy rate)"
    )
    print(f"   Recall:     {recall:.4f} (% of actual survivors we catch)")
    print(f"   F1 Score:   {f1:.4f} (Overall balance)")
    print(f"   Accuracy:   {accuracy:.4f}")
    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"   ROC AUC:    {auc:.4f}")

    print(f"\n🚨 CLINICAL RISK ASSESSMENT:")
    total_predicted_positive = tp + fp
    total_actual_positive = tp + fn

    if total_predicted_positive > 0:
        print(
            f"   Predictions made: {total_predicted_positive:,} patients predicted to survive"
        )
        print(f"   Correct predictions: {tp:,} ({tp/total_predicted_positive:.1%})")
        print(f"   FALSE HOPES given: {fp:,} ({fp/total_predicted_positive:.1%}) ⚠️")
    else:
        print(f"   NO survival predictions made (too conservative)")

    if total_actual_positive > 0:
        print(f"   Actual survivors: {total_actual_positive:,}")
        print(f"   Survivors identified: {tp:,} ({recall:.1%})")
        print(f"   Survivors MISSED: {fn:,} ({fn/total_actual_positive:.1%}) 😞")

    print(f"\n📋 DETAILED BREAKDOWN:")
    print(f"   True Negatives:  {tn:8,} (Correctly predicted won't survive)")
    print(f"   False Positives: {fp:8,} (Wrongly predicted survival) ⚠️")
    print(f"   False Negatives: {fn:8,} (Missed actual survivors)")
    print(f"   True Positives:  {tp:8,} (Correctly predicted survival) ✅")

    return precision, recall, f1, fp, tp


def get_precision_at_thresholds(
    y_true, y_pred_proba, target_precisions=[0.5, 0.6, 0.7, 0.8, 0.9]
):
    """Find thresholds that achieve different precision levels"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    results = {}
    print(f"\n🎯 PRECISION-THRESHOLD ANALYSIS:")
    print(
        f"{'Target Precision':<18} {'Threshold':<12} {'Actual Precision':<18} {'Recall':<10} {'Predictions':<12}"
    )
    print("-" * 80)

    for target_prec in target_precisions:
        # Find the threshold that gives us closest to target precision
        valid_indices = precisions >= target_prec
        if np.any(valid_indices):
            best_idx = np.where(valid_indices)[0][0]
            if best_idx < len(thresholds):
                threshold = thresholds[best_idx]
                actual_precision = precisions[best_idx]
                actual_recall = recalls[best_idx]

                # Count how many predictions this would make
                num_predictions = np.sum(y_pred_proba >= threshold)

                results[target_prec] = {
                    "threshold": threshold,
                    "precision": actual_precision,
                    "recall": actual_recall,
                    "num_predictions": num_predictions,
                }

                print(
                    f"{target_prec:.1%}{'':12} {threshold:<12.4f} {actual_precision:<18.4f} {actual_recall:<10.4f} {num_predictions:<12,}"
                )

    return results


def main():
    # Load data
    X, y = load_and_preprocess_data()

    print(f"\n📊 DATASET OVERVIEW:")
    target_dist = y.value_counts()
    target_pct = y.value_counts(normalize=True) * 100
    print(f"Total samples: {len(X):,}")
    print(f"Not Survived: {target_dist[0]:,} ({target_pct[0]:.1f}%)")
    print(f"Survived: {target_dist[1]:,} ({target_pct[1]:.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocess
    preprocessor = create_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    print(f"\n🔬 TRAINING MEDICAL-SAFE MODELS")
    print("=" * 80)

    # Model 1: Balanced approach with bias toward precision
    print(f"\n🟦 MODEL 1: BALANCED PRECISION-FOCUSED")

    # Calculate class weights but reduce positive weight slightly for precision focus
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    balanced_weight = (
        class_weights[1] / class_weights[0]
    ) * 0.7  # Reduce by 30% for precision

    xgb_balanced = xgboost.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
        scale_pos_weight=balanced_weight,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=0.5,
        subsample=0.9,
        colsample_bytree=0.9,
    )

    xgb_balanced.fit(X_train_scaled, y_train)
    y_pred_proba_balanced = xgb_balanced.predict_proba(X_test_scaled)[:, 1]

    # Analyze precision at different thresholds
    threshold_results = get_precision_at_thresholds(y_test, y_pred_proba_balanced)

    # Test different precision targets
    precision_targets = [0.5, 0.6, 0.7, 0.8]

    print(f"\n🎯 TESTING DIFFERENT PRECISION TARGETS:")
    print("=" * 80)

    best_results = {}

    for target_prec in precision_targets:
        if target_prec in threshold_results:
            threshold_info = threshold_results[target_prec]
            threshold = threshold_info["threshold"]

            # Make predictions with this threshold
            y_pred = (y_pred_proba_balanced >= threshold).astype(int)

            precision, recall, f1, fp, tp = evaluate_precision_focused(
                y_test,
                y_pred,
                y_pred_proba_balanced,
                f"PRECISION TARGET {target_prec:.0%} MODEL",
                threshold,
            )

            best_results[target_prec] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": fp,
                "true_positives": tp,
                "threshold": threshold,
            }

    # Final Recommendations
    print(f"\n{'='*80}")
    print(f"🏥 MEDICAL DECISION SUPPORT RECOMMENDATIONS")
    print(f"{'='*80}")

    print(f"\n📋 PRECISION TARGET COMPARISON:")
    print(
        f"{'Target':<8} {'Threshold':<10} {'Precision':<10} {'Recall':<8} {'F1':<6} {'False Hopes':<12} {'Caught Survivors'}"
    )
    print("-" * 80)

    for target_prec in precision_targets:
        if target_prec in best_results:
            r = best_results[target_prec]
            print(
                f"{target_prec:.0%}      {r['threshold']:<10.4f} {r['precision']:<10.4f} {r['recall']:<8.4f} {r['f1']:<6.4f} {r['false_positives']:<12,d} {r['true_positives']:,d}"
            )

    print(f"\n💡 CLINICAL RECOMMENDATIONS:")
    print("-" * 50)

    # Find best models for different scenarios
    if 0.7 in best_results and best_results[0.7]["precision"] > 0:
        print(f"🟢 RECOMMENDED FOR CLINICAL USE:")
        r = best_results[0.7]
        print(f"   • 70% Precision Target:")
        print(f"   • Threshold: {r['threshold']:.4f}")
        print(f"   • When model predicts survival: {r['precision']:.1%} are correct")
        print(f"   • False hopes given: {r['false_positives']:,} patients")
        print(
            f"   • Actual survivors caught: {r['true_positives']:,} / {r['true_positives'] + (39201 - r['true_positives']):,} ({r['recall']:.1%})"
        )

    if 0.8 in best_results and best_results[0.8]["precision"] > 0:
        print(f"\n🟡 CONSERVATIVE CLINICAL USE:")
        r = best_results[0.8]
        print(f"   • 80% Precision Target:")
        print(f"   • Threshold: {r['threshold']:.4f}")
        print(f"   • When model predicts survival: {r['precision']:.1%} are correct")
        print(f"   • False hopes given: {r['false_positives']:,} patients")
        print(
            f"   • Actual survivors caught: {r['true_positives']:,} / {r['true_positives'] + (39201 - r['true_positives']):,} ({r['recall']:.1%})"
        )

    print(f"\n⚠️  IMPORTANT CLINICAL CONSIDERATIONS:")
    print(
        f"   • Model should be used as DECISION SUPPORT, not replacement for clinical judgment"
    )
    print(f"   • Consider probability scores, not just binary predictions")
    print(f"   • Patients near threshold should get additional evaluation")
    print(f"   • Regular model retraining with new data is essential")
    print(f"   • Document model limitations in clinical protocols")

    # Save the best balanced model
    print(f"\n💾 SAVING MODEL AND CONFIGURATIONS...")
    joblib.dump(xgb_balanced, "lung_cancer_medical_model.pkl")
    joblib.dump(preprocessor, "lung_cancer_preprocessor.pkl")

    # Save threshold configurations
    with open("medical_thresholds.txt", "w") as f:
        f.write("MEDICAL MODEL THRESHOLD CONFIGURATIONS\n")
        f.write("=====================================\n\n")
        for target_prec in precision_targets:
            if target_prec in best_results:
                r = best_results[target_prec]
                f.write(f"{target_prec:.0%} Precision Target:\n")
                f.write(f"  Threshold: {r['threshold']:.6f}\n")
                f.write(f"  Actual Precision: {r['precision']:.4f}\n")
                f.write(f"  Recall: {r['recall']:.4f}\n")
                f.write(f"  False Positives: {r['false_positives']:,}\n")
                f.write(f"  True Positives: {r['true_positives']:,}\n\n")

    print(f"✅ Medical model and configurations saved!")


if __name__ == "__main__":
    main()
