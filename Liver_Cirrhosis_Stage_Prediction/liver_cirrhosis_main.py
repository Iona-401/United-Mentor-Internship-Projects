import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import shap
import os

FILE_PATH = "liver_cirrhosis.csv"  # CSV file in current directory


class EnhancedLiverCirrhosisPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.feature_names = None
        self.scaler = None
        self.explainer = None
        self.preprocessor = None

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the liver cirrhosis dataset"""
        print("📊 Loading and preprocessing data...")

        # Load data
        data = pd.read_csv(file_path)
        print(f"Dataset shape: {data.shape}")
        print(f"Missing values per column:\n{data.isnull().sum()}")

        # Handle missing values
        data.dropna(inplace=True)
        print(f"After removing missing values: {data.shape}")

        # Map categorical variables to numerical
        mapping_dict = {
            "Status": {"C": 0, "CL": 1, "D": 2},
            "Drug": {"D-penicillamine": 1, "Placebo": 0},
            "Sex": {"M": 1, "F": 0},
            "Ascites": {"Y": 1, "N": 0},
            "Hepatomegaly": {"Y": 1, "N": 0},
            "Spiders": {"Y": 1, "N": 0},
            "Edema": {"Y": 2, "S": 1, "N": 0},
            "Stage": {1: 0, 2: 1, 3: 2, 4: 3},  # Updated to handle 4 stages if present
        }

        for col, mapping in mapping_dict.items():
            if col in data.columns:
                data[col] = data[col].map(mapping)

        # Separate features and target
        X = data.drop("Stage", axis=1)
        y = data["Stage"]

        print(f"Target distribution:\n{y.value_counts().sort_index()}")

        return X, y, data

    def setup_preprocessing(self, X):
        """Setup preprocessing pipelines"""
        print("🔧 Setting up preprocessing pipelines...")

        # Define feature types
        numerical_features = [
            "N_Days",
            "Age",
            "Bilirubin",
            "Cholesterol",
            "Albumin",
            "Copper",
            "Alk_Phos",
            "SGOT",
            "Tryglicerides",
            "Platelets",
            "Prothrombin",
        ]
        categorical_features = [
            "Status",
            "Drug",
            "Sex",
            "Ascites",
            "Hepatomegaly",
            "Spiders",
            "Edema",
        ]

        # Filter features that actually exist in the dataset
        numerical_features = [f for f in numerical_features if f in X.columns]
        categorical_features = [f for f in categorical_features if f in X.columns]

        print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
        print(
            f"Categorical features ({len(categorical_features)}): {categorical_features}"
        )

        # Create transformers
        numeric_transformer = Pipeline([("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]
        )

        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Fit preprocessor to get feature names
        self.preprocessor.fit(X)

        # Get feature names after preprocessing
        num_feature_names = numerical_features
        cat_feature_names = (
            self.preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_features)
        )
        self.feature_names = num_feature_names + list(cat_feature_names)

        return self.preprocessor

    def benchmark_models(self, X, y):
        """Compare multiple ML algorithms with cross-validation"""
        print("\n🏆 Benchmarking Multiple Models...")

        # Define models to compare
        models_config = {
            "Random Forest": RandomForestClassifier(
                n_estimators=500, max_depth=None, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, random_state=42
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
            ),
            "Logistic Regression": LogisticRegression(
                random_state=42, max_iter=1000, multi_class="ovr"
            ),
            "SVM": SVC(random_state=42, probability=True, kernel="rbf"),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
            ),
        }

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}

        for name, model in models_config.items():
            print(f"\n🔄 Training {name}...")

            # Create pipeline
            pipeline = Pipeline(
                [("preprocessor", self.preprocessor), ("classifier", model)]
            )

            # Cross-validation scores
            try:
                cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
                cv_f1_scores = cross_val_score(
                    pipeline, X, y, cv=cv, scoring="f1_weighted"
                )

                results[name] = {
                    "accuracy_mean": cv_scores.mean(),
                    "accuracy_std": cv_scores.std(),
                    "f1_mean": cv_f1_scores.mean(),
                    "f1_std": cv_f1_scores.std(),
                    "pipeline": pipeline,
                }

                print(
                    f"✅ {name} - Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})"
                )
                print(
                    f"   F1-Score: {cv_f1_scores.mean():.4f} (±{cv_f1_scores.std():.4f})"
                )

            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
                continue

        # Select best model based on F1-score
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]["f1_mean"])
            self.best_model = results[best_model_name]["pipeline"]
            self.best_score = results[best_model_name]["f1_mean"]

            print(
                f"\n🥇 Best Model: {best_model_name} (F1-Score: {self.best_score:.4f})"
            )

        return results, best_model_name if results else None

    def detailed_evaluation(self, X, y):
        """Comprehensive model evaluation with train-test split"""
        print("\n📈 Detailed Model Evaluation...")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit the best model
        self.best_model.fit(X_train, y_train)

        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Create evaluation plots
        self.plot_evaluation_metrics(y_test, y_pred, y_pred_proba)

        return X_train, X_test, y_train, y_test

    def plot_evaluation_metrics(self, y_test, y_pred, y_pred_proba):
        """Create comprehensive evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[0, 0], cmap="Blues")
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_ylabel("Actual Stage")
        axes[0, 0].set_xlabel("Predicted Stage")

        # Class Distribution
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        test_counts = [np.sum(y_test == cls) for cls in unique_classes]
        pred_counts = [np.sum(y_pred == cls) for cls in unique_classes]

        x = np.arange(len(unique_classes))
        width = 0.35

        axes[0, 1].bar(x - width / 2, test_counts, width, label="Actual", alpha=0.8)
        axes[0, 1].bar(x + width / 2, pred_counts, width, label="Predicted", alpha=0.8)
        axes[0, 1].set_title("Class Distribution Comparison")
        axes[0, 1].set_xlabel("Cirrhosis Stage")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f"Stage {cls}" for cls in unique_classes])
        axes[0, 1].legend()

        # Feature Importance (if available)
        if hasattr(self.best_model.named_steps["classifier"], "feature_importances_"):
            importances = self.best_model.named_steps["classifier"].feature_importances_
            indices = np.argsort(importances)[::-1][:15]

            axes[1, 0].bar(range(len(indices)), importances[indices])
            axes[1, 0].set_title("Top 15 Feature Importances")
            axes[1, 0].set_xticks(range(len(indices)))
            axes[1, 0].set_xticklabels(
                [self.feature_names[i] for i in indices], rotation=45
            )
            axes[1, 0].set_ylabel("Importance")

        # Model Performance by Stage
        stages = np.unique(y_test)
        stage_accuracies = []

        for stage in stages:
            stage_mask = y_test == stage
            if np.sum(stage_mask) > 0:
                stage_acc = accuracy_score(y_test[stage_mask], y_pred[stage_mask])
                stage_accuracies.append(stage_acc)
            else:
                stage_accuracies.append(0)

        axes[1, 1].bar(range(len(stages)), stage_accuracies)
        axes[1, 1].set_title("Accuracy by Cirrhosis Stage")
        axes[1, 1].set_xlabel("Cirrhosis Stage")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_xticks(range(len(stages)))
        axes[1, 1].set_xticklabels([f"Stage {stage}" for stage in stages])
        axes[1, 1].set_ylim(0, 1)

        # Add accuracy values on bars
        for i, acc in enumerate(stage_accuracies):
            axes[1, 1].text(i, acc + 0.02, f"{acc:.3f}", ha="center")

        plt.tight_layout()
        plt.savefig(
            "evaluation_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("📊 Evaluation plots saved to 'evaluation_metrics.png'")

    def setup_explainability(self, X_train, X_test):
        """Setup SHAP explainer for model interpretability"""
        print("\n🔍 Setting up Model Explainability...")

        try:
            # Get the final classifier from pipeline
            classifier = self.best_model.named_steps["classifier"]
            preprocessor = self.best_model.named_steps["preprocessor"]

            # Transform training data for SHAP
            X_train_transformed = preprocessor.transform(X_train)

            # Create SHAP explainer
            if hasattr(classifier, "predict_proba"):
                # Use a sample for efficiency
                sample_size = min(100, len(X_train_transformed))
                self.explainer = shap.Explainer(
                    classifier, X_train_transformed[:sample_size]
                )
                print("✅ SHAP explainer ready for predictions")

        except Exception as e:
            print(f"⚠️ SHAP setup failed: {e}")
            self.explainer = None

        return self.explainer

    def explain_prediction(self, sample_data):
        """Generate SHAP explanation for a single prediction"""
        if self.explainer is None:
            print("❌ Explainer not initialized.")
            return None

        try:
            # Transform the sample using the preprocessor
            preprocessor = self.best_model.named_steps["preprocessor"]
            sample_transformed = preprocessor.transform([sample_data])

            # Get SHAP values
            shap_values = self.explainer.shap_values(sample_transformed)

            # Handle multiclass output
            if isinstance(shap_values, list):
                # For multiclass, show the prediction class SHAP values
                prediction = self.best_model.predict([sample_data])[0]
                shap_values = shap_values[prediction]

            return shap_values[0] if len(shap_values.shape) > 1 else shap_values

        except Exception as e:
            print(f"❌ Error generating explanation: {e}")
            return None

    def save_models(self, output_dir="."):
        """Save all models and components"""
        print(f"\n💾 Saving models to {output_dir}...")

        os.makedirs(output_dir, exist_ok=True)

        # Save the best model pipeline (this includes all preprocessing)
        joblib.dump(self.best_model, f"{output_dir}/best_liver_cirrhosis_model.pkl")

        # Also save with the specific name the app expects
        joblib.dump(
            self.best_model, f"{output_dir}/random_forest_liver_cirrhosis_model.pkl"
        )

        # Save feature names
        joblib.dump(self.feature_names, f"{output_dir}/feature_names.pkl")

        # Extract and save the scaler from the preprocessing pipeline
        try:
            if self.preprocessor is not None:
                # Get the scaler from the numeric transformer in the column transformer
                numeric_transformer = self.preprocessor.named_transformers_["num"]
                scaler = numeric_transformer.named_steps["scaler"]
                joblib.dump(scaler, f"{output_dir}/scaler_liver_cirrhosis.pkl")
                print("✅ Scaler extracted and saved from preprocessing pipeline")
            else:
                # Create a simple scaler as fallback
                from sklearn.preprocessing import StandardScaler

                fallback_scaler = StandardScaler()
                joblib.dump(fallback_scaler, f"{output_dir}/scaler_liver_cirrhosis.pkl")
                print("⚠️ Created fallback scaler (not fitted)")
        except Exception as e:
            print(f"⚠️ Could not extract scaler: {e}")
            # Create a simple scaler as fallback
            from sklearn.preprocessing import StandardScaler

            fallback_scaler = StandardScaler()
            joblib.dump(fallback_scaler, f"{output_dir}/scaler_liver_cirrhosis.pkl")

        # Save explainer if available
        if self.explainer:
            joblib.dump(self.explainer, f"{output_dir}/shap_explainer.pkl")

        # Save the preprocessing pipeline separately for the app
        if self.preprocessor:
            joblib.dump(self.preprocessor, f"{output_dir}/preprocessor.pkl")

        print("✅ All models and components saved successfully!")


def main():
    """Main execution function"""
    print("🏥 Enhanced Liver Cirrhosis Stage Prediction System")
    print("=" * 60)

    # Initialize predictor
    predictor = EnhancedLiverCirrhosisPredictor()

    # Load and preprocess data
    X, y, data = predictor.load_and_preprocess_data(FILE_PATH)

    # Setup preprocessing
    predictor.setup_preprocessing(X)

    # Benchmark models
    results, best_model_name = predictor.benchmark_models(X, y)

    if best_model_name:
        # Detailed evaluation
        X_train, X_test, y_train, y_test = predictor.detailed_evaluation(X, y)

        # Setup explainability
        predictor.setup_explainability(X_train, X_test)

        # Save everything
        predictor.save_models()

        print(f"\n🎉 Enhanced Liver Cirrhosis Prediction System Complete!")
        print(f"Best Model: {best_model_name}")
        print(f"Best F1-Score: {predictor.best_score:.4f}")
    else:
        print("❌ No models were successfully trained.")


if __name__ == "__main__":
    main()
