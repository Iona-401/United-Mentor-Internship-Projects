import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

FILE_PATH = "Lung Cancer/dataset_med.csv"


def analyze_dataset_naturalness():
    """Analyze if the dataset has realistic patterns"""
    print("🔍 DATASET NATURALNESS ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(FILE_PATH)
    print(f"Dataset shape: {df.shape}")

    # Drop unnecessary columns for analysis
    columns_to_drop = ["id", "country", "diagnosis_date", "end_treatment_date"]
    df_clean = df.drop(columns=columns_to_drop)

    # 1. Check if features actually correlate with survival (they should!)
    print(f"\n📊 CORRELATION ANALYSIS:")
    print("-" * 40)

    # Map categorical to numeric for correlation
    df_numeric = df_clean.copy()

    # Convert categorical to numeric
    categorical_cols = df_numeric.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if col != "survived":
            df_numeric[col] = pd.Categorical(df_numeric[col]).codes

    correlations = df_numeric.corr()["survived"].sort_values(key=abs, ascending=False)
    print("Correlations with survival (should show some meaningful patterns):")
    for feature, corr in correlations.items():
        if feature != "survived":
            print(f"  {feature:<20}: {corr:>8.4f}")

    # 2. Check for suspicious uniform distributions
    print(f"\n🎲 DISTRIBUTION ANALYSIS:")
    print("-" * 40)

    for col in ["age", "bmi", "cholesterol_level"]:
        data = df_clean[col]
        # Test for uniform distribution
        _, p_value = stats.kstest(
            data, "uniform", args=(data.min(), data.max() - data.min())
        )
        print(f"{col}:")
        print(f"  Range: {data.min():.1f} - {data.max():.1f}")
        print(f"  Mean: {data.mean():.2f}, Std: {data.std():.2f}")
        print(f"  Uniform distribution test p-value: {p_value:.6f}")
        if p_value > 0.05:
            print(f"  ⚠️ WARNING: May be artificially uniform!")
        print()

    # 3. Check for realistic medical relationships
    print(f"🏥 MEDICAL LOGIC CHECKS:")
    print("-" * 40)

    # Age vs Survival (older patients typically have worse outcomes)
    age_survival = df_clean.groupby("survived")["age"].mean()
    print(f"Average age by survival status:")
    print(f"  Not survived: {age_survival[0]:.1f} years")
    print(f"  Survived: {age_survival[1]:.1f} years")
    if age_survival[0] > age_survival[1]:
        print("  ✅ Expected: Non-survivors are older")
    else:
        print("  ⚠️ Unexpected: Survivors are older (unusual)")

    # Cancer stage vs survival (advanced stages should have worse survival)
    stage_survival = (
        df_clean.groupby(["cancer_stage", "survived"]).size().unstack(fill_value=0)
    )
    stage_survival_rate = stage_survival[1] / (stage_survival[0] + stage_survival[1])
    print(f"\nSurvival rates by cancer stage:")
    for stage, rate in stage_survival_rate.items():
        print(f"  {stage}: {rate:.3f} ({rate:.1%})")

    # Check if later stages have lower survival (they should)
    stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    if all(stage in stage_survival_rate.index for stage in stages):
        stage_rates = [stage_survival_rate[stage] for stage in stages]
        is_decreasing = all(
            stage_rates[i] >= stage_rates[i + 1] for i in range(len(stage_rates) - 1)
        )
        if is_decreasing:
            print("  ✅ Expected: Later stages have worse survival")
        else:
            print("  ⚠️ Unexpected: Survival rates don't decrease with stage")

    # 4. Check for suspiciously balanced categorical distributions
    print(f"\n⚖️ CATEGORICAL BALANCE CHECK:")
    print("-" * 40)

    categorical_features = [
        "gender",
        "family_history",
        "smoking_status",
        "treatment_type",
    ]
    for feature in categorical_features:
        if feature in df_clean.columns:
            value_counts = df_clean[feature].value_counts()
            print(f"{feature}:")
            for value, count in value_counts.items():
                percentage = count / len(df_clean) * 100
                print(f"  {value}: {count:,} ({percentage:.1f}%)")

            # Check if suspiciously balanced
            if len(value_counts) > 1:
                max_pct = max(value_counts) / len(df_clean) * 100
                min_pct = min(value_counts) / len(df_clean) * 100
                if abs(max_pct - min_pct) < 5:  # Within 5% of each other
                    print(f"  ⚠️ WARNING: Suspiciously balanced distribution!")
            print()

    # 5. Check for impossible medical values
    print(f"🚨 MEDICAL VALIDITY CHECKS:")
    print("-" * 40)

    # Age checks
    impossible_ages = df_clean[(df_clean["age"] < 0) | (df_clean["age"] > 120)]
    print(f"Impossible ages (< 0 or > 120): {len(impossible_ages)}")

    # BMI checks
    impossible_bmi = df_clean[(df_clean["bmi"] < 10) | (df_clean["bmi"] > 80)]
    print(f"Extreme BMI values (< 10 or > 80): {len(impossible_bmi)}")

    # Cholesterol checks
    impossible_chol = df_clean[
        (df_clean["cholesterol_level"] < 100) | (df_clean["cholesterol_level"] > 500)
    ]
    print(f"Extreme cholesterol (< 100 or > 500): {len(impossible_chol)}")

    # 6. Pattern recognition issues
    print(f"\n🤖 ML PATTERN ISSUES:")
    print("-" * 40)

    # Check if survival is truly random
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Prepare data for quick ML test
    X = df_clean.drop("survived", axis=1)
    y = df_clean["survived"]

    # Encode categorical variables
    le_dict = {}
    for col in X.select_dtypes(include=["object"]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    # Quick random forest to test predictability
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)

    print(f"Random Forest performance:")
    print(f"  Training accuracy: {train_score:.4f}")
    print(f"  Test accuracy: {test_score:.4f}")
    print(f"  Baseline (majority class): {max(y.value_counts()) / len(y):.4f}")

    improvement = test_score - (max(y.value_counts()) / len(y))
    if improvement < 0.05:
        print(
            f"  ⚠️ WARNING: Model barely beats baseline! Improvement: {improvement:.4f}"
        )
        print(f"  This suggests features have little predictive power")

    # 7. Final assessment
    print(f"\n🎯 DATASET NATURALNESS ASSESSMENT:")
    print("-" * 50)

    issues_found = []

    # Check correlation strength
    max_abs_corr = max(
        abs(corr) for corr in correlations.values if corr != correlations["survived"]
    )
    if max_abs_corr < 0.1:
        issues_found.append("Very weak feature-target correlations")

    # Check ML improvement
    if improvement < 0.05:
        issues_found.append("ML models can't find meaningful patterns")

    # Check medical logic
    if not (age_survival[0] > age_survival[1]):
        issues_found.append("Age-survival relationship is counter-intuitive")

    if len(issues_found) == 0:
        print("✅ Dataset appears to have natural patterns")
    else:
        print("❌ Dataset has concerning artificial characteristics:")
        for issue in issues_found:
            print(f"  • {issue}")

        print(f"\n💡 LIKELY EXPLANATIONS:")
        print("  • Synthetic/artificially generated data")
        print("  • Data heavily anonymized/scrambled")
        print("  • Missing crucial predictive features")
        print("  • Survival outcomes may be truly random in this dataset")
        print("  • Data quality issues or collection problems")


if __name__ == "__main__":
    analyze_dataset_naturalness()
