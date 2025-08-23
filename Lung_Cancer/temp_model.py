import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Load the data
FILE_PATH = "Lung Cancer\dataset_med.csv"

print("Loading Lung Cancer Dataset for Undersampling Experiment...")
df = pd.read_csv(FILE_PATH)
print(f"Original dataset shape: {df.shape}")

# Same preprocessing as before
print(f"\nPreprocessing data...")
columns_to_drop = ["id", "country", "diagnosis_date", "end_treatment_date"]
df_clean = df.drop(columns=columns_to_drop)

# Handle missing values
df_clean.dropna(inplace=True)
print(f"After cleaning: {df_clean.shape}")

# Check original class distribution
print(f"\nOriginal Target Distribution:")
target_dist = df_clean['survived'].value_counts()
target_pct = df_clean['survived'].value_counts(normalize=True) * 100
print(f"Not Survived (0): {target_dist[0]:,} samples ({target_pct[0]:.1f}%)")
print(f"Survived (1): {target_dist[1]:,} samples ({target_pct[1]:.1f}%)")
print(f"Class imbalance ratio: {target_pct[0]/target_pct[1]:.2f}:1")

# Manual mapping of categorical variables
df_clean["gender"] = df_clean["gender"].map({"Male": 0, "Female": 1})
df_clean["family_history"] = df_clean["family_history"].map({"No": 0, "Yes": 1})
df_clean["cancer_stage"] = df_clean["cancer_stage"].map({
    "Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3
})
df_clean["treatment_type"] = df_clean["treatment_type"].map({
    "Surgery": 0, "Chemotherapy": 1, "Radiation": 2, "Immunotherapy": 3
})
df_clean["smoking_status"] = df_clean["smoking_status"].map({
    "Never Smoked": 0, "Former Smoker": 1, "Current Smoker": 2
})

# Check for unmapped values
unmapped_check = df_clean.isnull().sum()
if unmapped_check.sum() > 0:
    print(f"Removing {unmapped_check.sum()} rows with unmapped categorical values...")
    df_clean.dropna(inplace=True)

print(f"Final preprocessed shape: {df_clean.shape}")

# ============ UNDERSAMPLING EXPERIMENT ============
print(f"\n" + "="*60)
print("UNDERSAMPLING EXPERIMENT - CREATING TRUE 50/50 DATASET")
print("="*60)

# Separate classes
class_0 = df_clean[df_clean['survived'] == 0]  # Not survived
class_1 = df_clean[df_clean['survived'] == 1]  # Survived

print(f"Class 0 (Not Survived): {len(class_0):,} samples")
print(f"Class 1 (Survived): {len(class_1):,} samples")

# Undersample majority class to match minority class
n_minority = len(class_1)
print(f"\nUndersampling majority class to {n_minority:,} samples...")

# Randomly sample from majority class
class_0_undersampled = class_0.sample(n=n_minority, random_state=42)

# Combine classes to create balanced dataset
df_balanced = pd.concat([class_0_undersampled, class_1], axis=0)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"Balanced dataset shape: {df_balanced.shape}")
print(f"New class distribution:")
balanced_dist = df_balanced['survived'].value_counts()
balanced_pct = df_balanced['survived'].value_counts(normalize=True) * 100
print(f"Not Survived (0): {balanced_dist[0]:,} samples ({balanced_pct[0]:.1f}%)")
print(f"Survived (1): {balanced_dist[1]:,} samples ({balanced_pct[1]:.1f}%)")

# Prepare features and target from balanced dataset
X_balanced = df_balanced.drop("survived", axis=1)
y_balanced = df_balanced["survived"]

print(f"Balanced feature matrix shape: {X_balanced.shape}")

# Split balanced dataset
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print(f"Balanced training set: {X_train_bal.shape}")
print(f"Balanced test set: {X_test_bal.shape}")
print(f"Balanced training distribution: {pd.Series(y_train_bal).value_counts(normalize=True) * 100}")

# ============ MODEL COMPARISON ============
print(f"\n" + "="*60)
print("TRAINING MODELS ON DIFFERENT DATASETS")
print("="*60)

# 1. Original imbalanced data model (for comparison)
X_orig = df_clean.drop("survived", axis=1)
y_orig = df_clean["survived"]
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)

print(f"\n=== Model 1: Random Forest on Original Imbalanced Data ===")
rf_imbalanced = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1)
rf_imbalanced.fit(X_train_orig, y_train_orig)
y_pred_imbalanced = rf_imbalanced.predict(X_test_orig)
accuracy_imbalanced = accuracy_score(y_test_orig, y_pred_imbalanced)
f1_imbalanced = f1_score(y_test_orig, y_pred_imbalanced)
roc_auc_imbalanced = roc_auc_score(y_test_orig, rf_imbalanced.predict_proba(X_test_orig)[:, 1])

print(f"Imbalanced Data Model:")
print(f"  Accuracy: {accuracy_imbalanced:.4f} ({accuracy_imbalanced*100:.1f}%)")
print(f"  F1-Score: {f1_imbalanced:.4f}")
print(f"  ROC-AUC: {roc_auc_imbalanced:.4f}")
print(f"  Confusion Matrix:")
cm_imbalanced = confusion_matrix(y_test_orig, y_pred_imbalanced)
print(f"    [[{cm_imbalanced[0][0]:,}, {cm_imbalanced[0][1]:,}]")
print(f"     [{cm_imbalanced[1][0]:,}, {cm_imbalanced[1][1]:,}]]")

# Calculate survival prediction rate
survived_pred_imbalanced = sum(y_pred_imbalanced)
print(f"  Predicted Survivals: {survived_pred_imbalanced:,} ({survived_pred_imbalanced/len(y_test_orig)*100:.1f}%)")
print(f"  Actual Survivals: {sum(y_test_orig):,} ({sum(y_test_orig)/len(y_test_orig)*100:.1f}%)")

print(f"\n=== Model 2: Random Forest on True 50/50 Balanced Data ===")
rf_balanced = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)  # No class_weight needed
rf_balanced.fit(X_train_bal, y_train_bal)
y_pred_balanced = rf_balanced.predict(X_test_bal)
accuracy_balanced = accuracy_score(y_test_bal, y_pred_balanced)
f1_balanced = f1_score(y_test_bal, y_pred_balanced)
roc_auc_balanced = roc_auc_score(y_test_bal, rf_balanced.predict_proba(X_test_bal)[:, 1])

print(f"50/50 Balanced Data Model:")
print(f"  Accuracy: {accuracy_balanced:.4f} ({accuracy_balanced*100:.1f}%)")
print(f"  F1-Score: {f1_balanced:.4f}")
print(f"  ROC-AUC: {roc_auc_balanced:.4f}")
print(f"  Confusion Matrix:")
cm_balanced = confusion_matrix(y_test_bal, y_pred_balanced)
print(f"    [[{cm_balanced[0][0]:,}, {cm_balanced[0][1]:,}]")
print(f"     [{cm_balanced[1][0]:,}, {cm_balanced[1][1]:,}]]")

survived_pred_balanced = sum(y_pred_balanced)
print(f"  Predicted Survivals: {survived_pred_balanced:,} ({survived_pred_balanced/len(y_test_bal)*100:.1f}%)")
print(f"  Actual Survivals: {sum(y_test_bal):,} ({sum(y_test_bal)/len(y_test_bal)*100:.1f}%)")

print(f"\n=== Model 3: Ensemble on 50/50 Balanced Data ===")
# Create ensemble for balanced data
rf_ens = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
gb_ens = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42)
et_ens = ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42)

ensemble_balanced = VotingClassifier(
    estimators=[('rf', rf_ens), ('gb', gb_ens), ('et', et_ens)],
    voting='soft'
)

ensemble_balanced.fit(X_train_bal, y_train_bal)
y_pred_ensemble = ensemble_balanced.predict(X_test_bal)
accuracy_ensemble = accuracy_score(y_test_bal, y_pred_ensemble)
f1_ensemble = f1_score(y_test_bal, y_pred_ensemble)
roc_auc_ensemble = roc_auc_score(y_test_bal, ensemble_balanced.predict_proba(X_test_bal)[:, 1])

print(f"Ensemble on Balanced Data:")
print(f"  Accuracy: {accuracy_ensemble:.4f} ({accuracy_ensemble*100:.1f}%)")
print(f"  F1-Score: {f1_ensemble:.4f}")
print(f"  ROC-AUC: {roc_auc_ensemble:.4f}")
print(f"  Confusion Matrix:")
cm_ensemble = confusion_matrix(y_test_bal, y_pred_ensemble)
print(f"    [[{cm_ensemble[0][0]:,}, {cm_ensemble[0][1]:,}]")
print(f"     [{cm_ensemble[1][0]:,}, {cm_ensemble[1][1]:,}]]")

survived_pred_ensemble = sum(y_pred_ensemble)
print(f"  Predicted Survivals: {survived_pred_ensemble:,} ({survived_pred_ensemble/len(y_test_bal)*100:.1f}%)")

# ============ RESULTS COMPARISON ============
print(f"\n" + "="*80)
print("EXPERIMENT RESULTS COMPARISON")
print("="*80)

print(f"{'Metric':<20} {'Imbalanced':<15} {'50/50 RF':<15} {'50/50 Ensemble':<15}")
print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")
print(f"{'Accuracy':<20} {accuracy_imbalanced*100:<15.1f} {accuracy_balanced*100:<15.1f} {accuracy_ensemble*100:<15.1f}")
print(f"{'F1-Score':<20} {f1_imbalanced:<15.3f} {f1_balanced:<15.3f} {f1_ensemble:<15.3f}")
print(f"{'ROC-AUC':<20} {roc_auc_imbalanced:<15.3f} {roc_auc_balanced:<15.3f} {roc_auc_ensemble:<15.3f}")

print(f"\nSurvival Prediction Analysis:")
print(f"{'Model':<20} {'Predicted':<12} {'Actual':<12} {'Difference':<12}")
print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")

imbal_actual = sum(y_test_orig)
imbal_pred = survived_pred_imbalanced
imbal_diff = abs(imbal_pred - imbal_actual)

bal_actual = sum(y_test_bal) 
bal_pred = survived_pred_balanced
bal_diff = abs(bal_pred - bal_actual)

ens_pred = survived_pred_ensemble
ens_diff = abs(ens_pred - bal_actual)  # Same test set as balanced

print(f"{'Imbalanced':<20} {imbal_pred:<12,} {imbal_actual:<12,} {imbal_diff:<12,}")
print(f"{'50/50 RF':<20} {bal_pred:<12,} {bal_actual:<12,} {bal_diff:<12,}")
print(f"{'50/50 Ensemble':<20} {ens_pred:<12,} {bal_actual:<12,} {ens_diff:<12,}")

# Determine best model
models = {
    'Imbalanced RF': (f1_imbalanced, roc_auc_imbalanced, accuracy_imbalanced),
    '50/50 RF': (f1_balanced, roc_auc_balanced, accuracy_balanced),
    '50/50 Ensemble': (f1_ensemble, roc_auc_ensemble, accuracy_ensemble)
}

# Use F1-score as primary metric for medical data
best_model_name = max(models.keys(), key=lambda k: models[k][0])
best_f1, best_roc_auc, best_acc = models[best_model_name]

print(f"\n" + "="*60)
print("CONCLUSION")
print("="*60)
print(f"Best Model (by F1-score): {best_model_name}")
print(f"F1-Score: {best_f1:.3f}")
print(f"ROC-AUC: {best_roc_auc:.3f}")
print(f"Accuracy: {best_acc:.1%}")

# Save the best model
if best_model_name == 'Imbalanced RF':
    best_model = rf_imbalanced
    model_data = "imbalanced"
elif best_model_name == '50/50 RF':
    best_model = rf_balanced
    model_data = "balanced"
else:
    best_model = ensemble_balanced
    model_data = "balanced_ensemble"

joblib.dump(best_model, f"lung_cancer_undersampled_{model_data}_model.pkl")

print(f"\nKey Insights:")
print(f"1. Undersampling approach feasibility: {'✅ Promising' if best_f1 > 0.3 else '❌ Limited'}")
print(f"2. F1-score improvement: {'✅ Significant' if best_f1 > f1_imbalanced + 0.1 else '⚠️ Moderate' if best_f1 > f1_imbalanced else '❌ None'}")
print(f"3. Clinical utility: {'✅ Better balance' if best_f1 > 0.4 else '⚠️ Still biased'}")
print(f"4. Recommended approach: {best_model_name}")

if model_data in ['balanced', 'balanced_ensemble']:
    print(f"\n✅ UNDERSAMPLING EXPERIMENT SUCCESSFUL!")
    print(f"   The 50/50 balanced dataset shows better performance metrics.")
    print(f"   This approach should be used for the final lung cancer model.")
else:
    print(f"\n⚠️  UNDERSAMPLING EXPERIMENT INCONCLUSIVE")
    print(f"   The original imbalanced approach still performs better.")
    print(f"   Consider other balancing techniques or feature engineering.")

print(f"\nModel saved as: lung_cancer_undersampled_{model_data}_model.pkl")
print(f"Ready for GUI application development!")
