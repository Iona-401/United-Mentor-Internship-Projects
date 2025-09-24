import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import shap

# Constants
DATASET_PATH = "Heart_Disease_Prediction/dataset.csv"


class HeartDiseasePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.best_model = None
        self.best_score = 0
        self.feature_names = None
        self.explainer = None

    def data_loader(self, data_path):
        """Load and preprocess the dataset."""
        print("Loading dataset...")
        data = pd.read_csv(data_path)
        print(f"Dataset Shape: {data.shape}")
        print(f"Class Distribution: \n{data["target"].value_counts()}")

        # Handle 0 values
        nonzero_mean = data.loc[data["cholesterol"] > 0, "cholesterol"].mean()
        data.loc[data["cholesterol"] == 0, "cholesterol"] = nonzero_mean
        print(f"Filled {(data["cholesterol"] == 0).sum()} zero cholesterol values")

        # Seperate Features
        X = data.drop("target", axis=1)
        y = data["target"]
        self.feature_names = X.columns.to_list()

        return X, y

    def benchmark_models(self, X, y):
        """Compare multiple Machine Learning Algorithms with cross-validation"""
        print("Starting model benchmarking...")
        # Define Models
        models_config = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42
            ),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        }

        # Cross-Validation
        cv = StratifiedKFold(shuffle=True, random_state=42)
        results = {}

        for name, model in models_config.items():
            print(f"\n Training {name}...")

            # Create Model Pipeline
            pipeline = ImbPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", model),
                ]
            )

            # Cross Validation Scores
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
            cv_roc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

            results[name] = {
                "accuracy_mean": cv_scores.mean(),
                "accuracy_std": cv_scores.std(),
                "roc_auc_mean": cv_roc_scores.mean(),
                "roc_auc_std": cv_roc_scores.std(),
                "pipeline": pipeline,
            }

            print(f"{name} - Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"ROC-AUC: {cv_roc_scores.mean():.4f} (±{cv_roc_scores.std():.4f})")

        # Select Best Model
        best_model_name = max(results.keys(), key=lambda k: results[k]["roc_auc_mean"])
        self.best_model = results[best_model_name]["pipeline"]
        self.best_score = results[best_model_name]["roc_auc_mean"]

        print(f"\nBest Model: {best_model_name} (ROC-AUC: {self.best_score:.4f})")

        return results, best_model_name

    def detailed_eval(self, X, y):
        """Comprehensive model evaluation"""
        print("Starting detailed evaluation...")
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit the best model
        self.best_model.fit(X_train, y_train)

        # Prediction
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Save evaluation plots
        self.plot_evaluation_metrics(y_test, y_pred, y_pred_proba)

        return X_train, X_test, y_train, y_test

    def plot_evaluation_metrics(self, y_test, y_pred, y_pred_proba):
        """Creates Comprehensive Eval Plots"""
        print("Creating evaluation plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[0, 0], cmap="Blues")
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_ylabel("Actual")
        axes[0, 0].set_xlabel("Predicted")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        axes[0, 1].plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].set_title("ROC Curve")
        axes[0, 1].legend(loc="lower right")

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1, 0].plot(recall, precision, color="blue", lw=2)
        axes[1, 0].set_xlabel("Recall")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Precision-Recall Curve")

        # Feature Importance (if Random Forest)
        if hasattr(self.best_model.named_steps["classifier"], "feature_importances_"):
            importances = self.best_model.named_steps["classifier"].feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            axes[1, 1].bar(range(len(indices)), importances[indices])
            axes[1, 1].set_title("Top 10 Feature Importances")
            axes[1, 1].set_xticks(range(len(indices)))
            axes[1, 1].set_xticklabels(
                [self.feature_names[i] for i in indices], rotation=45
            )

        plt.tight_layout()
        plt.savefig(
            "Heart_Disease_Prediction/evaluation_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Evaluation plots saved to 'evaluation_metrics.png'")

    def setup_explainability(self, X_train, y_train):
        """Setup SHAP explainer"""
        print("Setting up SHAP explainer...")

        # Get Final Classifier
        classifier = self.best_model.named_steps["classifier"]

        # Transform training Data for SHAP using the same pipeline
        scaler = self.best_model.named_steps["scaler"]
        X_train_scaled = scaler.transform(X_train)

        # Apply SMOTE to get the same transformation as in training
        smote = self.best_model.named_steps["smote"]
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_scaled, y_train
        )

        # Create SHAP explainer with resampled data (use a sample for efficiency)
        if hasattr(classifier, "predict_proba"):
            # Use a subset for efficiency (SHAP can be slow with large datasets)
            sample_size = min(100, len(X_train_resampled))
            self.explainer = shap.Explainer(classifier, X_train_resampled[:sample_size])
            print("SHAP explainer ready for predictions")

        return self.explainer

    def explain_prediction(self, sample_data, feature_names=None):
        """Generate SHAP explanation for a single prediction"""
        if self.explainer is None:
            print("Explainer not initialized. Run setup_explainability first.")
            return None

        # Transform the sample using the same pipeline steps
        scaler = self.best_model.named_steps["scaler"]
        sample_scaled = scaler.transform([sample_data])

        # Get SHAP values
        shap_values = self.explainer.shap_values(sample_scaled)

        # Create explanation plot
        if feature_names is None:
            feature_names = self.feature_names

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=sample_scaled[0],
                feature_names=feature_names,
            )
        )

        return shap_values[0]

    def save_models(self, output_dir="Heart_Disease_Prediction"):
        """Save best model"""
        print(f"\n Saving best model to {output_dir}")

        # Save the best model pipeline
        joblib.dump(self.best_model, f"{output_dir}/best_heart_disease_model.pkl")
        joblib.dump(self.feature_names, f"{output_dir}/feature_names.pkl")

        # Save explainer if available
        if self.explainer:
            joblib.dump(self.explainer, f"{output_dir}/shap_explainer.pkl")

        print("Models saved successfully!")


def main():
    print("Enhanced Heart Disease Prediction System")
    print("=" * 50)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Load and preprocess data
    X, y = predictor.data_loader(DATASET_PATH)

    # Benchmark models
    results, best_model_name = predictor.benchmark_models(X, y)

    # Detailed evaluation
    X_train, X_test, y_train, y_test = predictor.detailed_eval(X, y)

    # Setup explainability - pass y_train as well
    predictor.setup_explainability(X_train, y_train)

    # Save everything
    predictor.save_models()

    print("\nEnhanced Heart Disease Prediction System Complete!")
    print(f"Best Model: {best_model_name}")
    print(f"Best ROC-AUC Score: {predictor.best_score:.4f}")


if __name__ == "__main__":
    main()
