import sys
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import shap
import warnings

warnings.filterwarnings("ignore")

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QGroupBox,
    QTabWidget,
    QTextEdit,
    QScrollArea,
    QFrame,
    QSplitter,
    QProgressBar,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap


class PredictionWorker(QThread):
    """Background Thread for Model prediction and explanation"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, input_data, model, explainer, feature_names):
        super().__init__()
        self.input_data = input_data
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names

    def run(self):
        try:
            self.progress.emit(20)

            # Make Prediction
            prediction = self.model.predict([self.input_data])[0]
            probability = self.model.predict_proba([self.input_data])[0]

            self.progress.emit(50)

            # Generate SHAP Explanation
            shap_values = None
            if self.explainer is not None:
                try:
                    scaler = self.model.named_steps["scaler"]
                    sample_scaled = scaler.transform([self.input_data])
                    shap_vals = self.explainer.shap_values(sample_scaled)

                    # Handle different SHAP output formats
                    if isinstance(shap_vals, list):
                        # For binary classification, we might get a list
                        if len(shap_vals) > 1:
                            shap_values = shap_vals[1]  # Use positive class
                        else:
                            shap_values = shap_vals[0]
                    else:
                        shap_values = shap_vals

                    # Ensure we get a 1D array for the first sample
                    if hasattr(shap_values, "shape") and len(shap_values.shape) > 1:
                        shap_values = shap_values[0]

                    print(f"SHAP values shape: {np.array(shap_values).shape}")
                    print(
                        f"Feature names count: {len(self.feature_names) if self.feature_names else 'None'}"
                    )

                except Exception as e:
                    print(f"SHAP generation error: {e}")
                    shap_values = None

            self.progress.emit(100)

            # Emit Results
            result = {
                "prediction": prediction,
                "probability": probability,
                "shap_values": shap_values,
                "input_data": self.input_data,
                "feature_names": self.feature_names,
            }

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class PlotCanvas(FigureCanvas):
    """Custom matplotlib canvas for embedding plots"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_shap_waterfall(
        self, shap_values, expected_value, input_data, feature_names
    ):
        """Plot SHAP waterfall chart"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For binary classification, use the positive class (index 1) if available
            if len(shap_values) > 1:
                values = shap_values[1]  # Use positive class SHAP values
            else:
                values = shap_values[0]
        else:
            values = shap_values

        # Ensure values is a 1D array
        if hasattr(values, "shape") and len(values.shape) > 1:
            values = values[0]  # Get first sample if 2D

        # Convert to numpy array if not already
        values = np.array(values).flatten()

        # Ensure we have the right number of values
        if len(values) != len(feature_names):
            print(
                f"Warning: SHAP values length ({len(values)}) doesn't match features ({len(feature_names)})"
            )
            # Truncate or pad as needed
            min_len = min(len(values), len(feature_names))
            values = values[:min_len]
            feature_names = feature_names[:min_len]
            input_data = input_data[:min_len]

        features = feature_names
        data_values = input_data

        # Sort by absolute SHAP values
        sorted_idx = np.argsort(np.abs(values))[::-1][
            : min(10, len(values))
        ]  # Top 10 or all available

        # Prepare data for plotting
        y_pos = np.arange(len(sorted_idx))
        colors = ["#ff4444" if v < 0 else "#44ff44" for v in values[sorted_idx]]

        # Create horizontal bar plot
        bars = ax.barh(y_pos, values[sorted_idx], color=colors, alpha=0.7)

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{features[i]}: {data_values[i]:.2f}" for i in sorted_idx])
        ax.set_xlabel("SHAP Value (Impact on Prediction)")
        ax.set_title(
            "Feature Impact on Heart Disease Prediction", fontsize=14, fontweight="bold"
        )
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values[sorted_idx])):
            width = bar.get_width()
            ax.text(
                width + 0.01 if width >= 0 else width - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left" if width >= 0 else "right",
                va="center",
                fontweight="bold",
            )

        self.fig.tight_layout()
        self.draw()

    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance from the model"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if hasattr(model.named_steps["classifier"], "feature_importances_"):
            importances = model.named_steps["classifier"].feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
            bars = ax.bar(range(len(indices)), importances[indices], color=colors)

            ax.set_title(
                "Top 10 Feature Importances (Global)", fontsize=14, fontweight="bold"
            )
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels(
                [feature_names[i] for i in indices], rotation=45, ha="right"
            )
            ax.set_ylabel("Importance Score")

            # Add value labels on bars
            for bar, importance in zip(bars, importances[indices]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.001,
                    f"{importance:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        self.fig.tight_layout()
        self.draw()


class HeartDiseaseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.init_ui()
        self.load_models()
        self.apply_styles()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Enhanced Heart Disease Prediction System")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: Input and controls
        left_panel = self.create_input_panel()
        splitter.addWidget(left_panel)

        # Right panel: Results and explanations
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 1000])

    def create_input_panel(self):
        """Create the input panel with patient data fields"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(450)

        input_widget = QWidget()
        layout = QVBoxLayout(input_widget)

        # Title
        title = QLabel("Patient Information Input")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Input fields group
        input_group = QGroupBox("Clinical Parameters")
        input_layout = QGridLayout(input_group)

        # Define input fields with descriptions
        self.input_fields = {}
        fields_info = [
            ("age", "Age", "years", "40"),
            ("sex", "Sex", "1=Male, 0=Female", "1"),
            ("cp", "Chest Pain Type", "0-3", "1"),
            ("trestbps", "Resting Blood Pressure", "mmHg", "120"),
            ("chol", "Cholesterol", "mg/dl", "200"),
            ("fbs", "Fasting Blood Sugar", "1 if >120mg/dl", "0"),
            ("restecg", "Resting ECG", "0-2", "0"),
            ("thalach", "Max Heart Rate", "bpm", "150"),
            ("exang", "Exercise Induced Angina", "1=Yes, 0=No", "0"),
            ("oldpeak", "ST Depression", "0.0-6.2", "1.0"),
            ("slope", "Slope of ST Segment", "0-2", "1"),
        ]

        for i, (key, label, desc, default) in enumerate(fields_info):
            # Label
            lbl = QLabel(f"{label}:")
            lbl.setFont(QFont("Arial", 10, QFont.Bold))
            input_layout.addWidget(lbl, i, 0)

            # Input field
            field = QLineEdit()
            field.setText(default)
            field.setToolTip(f"{desc}")
            self.input_fields[key] = field
            input_layout.addWidget(field, i, 1)

            # Description
            desc_lbl = QLabel(desc)
            desc_lbl.setFont(QFont("Arial", 8))
            desc_lbl.setStyleSheet("color: #666666;")
            input_layout.addWidget(desc_lbl, i, 2)

        layout.addWidget(input_group)

        # Buttons
        button_layout = QVBoxLayout()

        self.predict_btn = QPushButton("🔍 Predict Heart Disease")
        self.predict_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.predict_btn.clicked.connect(self.predict)
        button_layout.addWidget(self.predict_btn)

        self.clear_btn = QPushButton("🔄 Clear Fields")
        self.clear_btn.clicked.connect(self.clear_fields)
        button_layout.addWidget(self.clear_btn)

        self.load_sample_btn = QPushButton("📝 Load Sample Data")
        self.load_sample_btn.clicked.connect(self.load_sample_data)
        button_layout.addWidget(self.load_sample_btn)

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        scroll.setWidget(input_widget)
        return scroll

    def create_results_panel(self):
        """Create the results panel with tabs for different views"""
        tab_widget = QTabWidget()

        # Prediction Results Tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        # Results display
        self.results_group = QGroupBox("Prediction Results")
        results_group_layout = QVBoxLayout(self.results_group)

        self.prediction_label = QLabel("No prediction yet")
        self.prediction_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        results_group_layout.addWidget(self.prediction_label)

        self.probability_label = QLabel("")
        self.probability_label.setFont(QFont("Arial", 12))
        self.probability_label.setAlignment(Qt.AlignCenter)
        results_group_layout.addWidget(self.probability_label)

        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Arial", 12))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        results_group_layout.addWidget(self.confidence_label)

        results_layout.addWidget(self.results_group)

        # Detailed explanation text
        self.explanation_text = QTextEdit()
        self.explanation_text.setMaximumHeight(200)
        self.explanation_text.setPlaceholderText(
            "Detailed explanation will appear here after prediction..."
        )
        results_layout.addWidget(self.explanation_text)

        tab_widget.addTab(results_tab, "🎯 Prediction Results")

        # SHAP Explanation Tab
        shap_tab = QWidget()
        shap_layout = QVBoxLayout(shap_tab)

        shap_label = QLabel("Individual Feature Impact (SHAP Analysis)")
        shap_label.setFont(QFont("Arial", 14, QFont.Bold))
        shap_label.setAlignment(Qt.AlignCenter)
        shap_layout.addWidget(shap_label)

        self.shap_canvas = PlotCanvas(width=10, height=8)
        shap_layout.addWidget(self.shap_canvas)

        tab_widget.addTab(shap_tab, "🔍 Feature Impact")

        # Feature Importance Tab
        importance_tab = QWidget()
        importance_layout = QVBoxLayout(importance_tab)

        importance_label = QLabel("Global Feature Importance")
        importance_label.setFont(QFont("Arial", 14, QFont.Bold))
        importance_label.setAlignment(Qt.AlignCenter)
        importance_layout.addWidget(importance_label)

        self.importance_canvas = PlotCanvas(width=10, height=8)
        importance_layout.addWidget(self.importance_canvas)

        tab_widget.addTab(importance_tab, "📊 Model Insights")

        return tab_widget

    def load_models(self):
        """Load the trained models and explainer"""
        try:
            model_path = "Heart_Disease_Prediction/best_heart_disease_model.pkl"
            feature_path = "Heart_Disease_Prediction/feature_names.pkl"
            explainer_path = "Heart_Disease_Prediction/shap_explainer.pkl"

            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.status_label.setText("✅ Model loaded successfully")
                print("Model loaded successfully")

            if os.path.exists(feature_path):
                self.feature_names = joblib.load(feature_path)
                print("Feature names loaded")

            if os.path.exists(explainer_path):
                self.explainer = joblib.load(explainer_path)
                print("SHAP explainer loaded")

            # Plot global feature importance
            if self.model and self.feature_names:
                self.importance_canvas.plot_feature_importance(
                    self.model, self.feature_names
                )

        except Exception as e:
            self.status_label.setText(f"❌ Error loading models: {str(e)}")
            print(f"Error loading models: {e}")

    def predict(self):
        """Make prediction with explanation"""
        if self.model is None:
            QMessageBox.warning(
                self, "Error", "Model not loaded. Please check model files."
            )
            return

        try:
            # Get input data
            input_data = []
            for key in [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
            ]:
                value = float(self.input_fields[key].text())
                input_data.append(value)

            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.predict_btn.setEnabled(False)

            # Start prediction in background thread
            self.worker = PredictionWorker(
                input_data, self.model, self.explainer, self.feature_names
            )
            self.worker.finished.connect(self.on_prediction_finished)
            self.worker.error.connect(self.on_prediction_error)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.start()

        except ValueError as e:
            QMessageBox.warning(
                self, "Input Error", f"Please enter valid numeric values: {str(e)}"
            )

    def on_prediction_finished(self, result):
        """Handle prediction results"""
        prediction = result["prediction"]
        probability = result["probability"]
        shap_values = result["shap_values"]
        input_data = result["input_data"]

        # Update prediction display
        if prediction == 1:
            self.prediction_label.setText("⚠️ HIGH RISK - Heart Disease Detected")
            self.prediction_label.setStyleSheet(
                "color: #ff4444; background-color: #ffeeee; padding: 10px; border-radius: 5px;"
            )
            risk_prob = probability[1] * 100
        else:
            self.prediction_label.setText("✅ LOW RISK - No Heart Disease Detected")
            self.prediction_label.setStyleSheet(
                "color: #44aa44; background-color: #eeffee; padding: 10px; border-radius: 5px;"
            )
            risk_prob = probability[0] * 100

        self.probability_label.setText(f"Confidence: {risk_prob:.1f}%")

        # Risk interpretation
        if risk_prob >= 90:
            confidence = "Very High Confidence"
        elif risk_prob >= 80:
            confidence = "High Confidence"
        elif risk_prob >= 70:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low Confidence"

        self.confidence_label.setText(confidence)

        # Generate detailed explanation
        self.generate_explanation(prediction, probability, input_data)

        # Plot SHAP explanation if available
        if shap_values is not None and self.feature_names:
            expected_value = self.explainer.expected_value if self.explainer else 0
            self.shap_canvas.plot_shap_waterfall(
                shap_values, expected_value, input_data, self.feature_names
            )

        # Reset UI
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        self.status_label.setText("Prediction completed")

    def on_prediction_error(self, error_msg):
        """Handle prediction errors"""
        QMessageBox.critical(
            self,
            "Prediction Error",
            f"An error occurred during prediction:\n{error_msg}",
        )
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        self.status_label.setText("Error occurred")

    def generate_explanation(self, prediction, probability, input_data):
        """Generate detailed medical explanation"""
        risk_factors = []
        protective_factors = []

        # Analyze key risk factors
        age = input_data[0]
        sex = input_data[1]
        cp = input_data[2]
        trestbps = input_data[3]
        chol = input_data[4]
        thalach = input_data[7]

        if age > 55:
            risk_factors.append(
                f"Advanced age ({age} years) increases cardiovascular risk"
            )
        if sex == 1 and age > 45:
            risk_factors.append("Male gender with age >45 is a risk factor")
        if cp in [2, 3]:
            risk_factors.append("Chest pain pattern suggests possible cardiac origin")
        if trestbps > 140:
            risk_factors.append(
                f"Elevated blood pressure ({trestbps} mmHg) indicates hypertension"
            )
        if chol > 240:
            risk_factors.append(
                f"High cholesterol ({chol} mg/dl) increases atherosclerosis risk"
            )
        if thalach < 100:
            risk_factors.append(
                "Low maximum heart rate may indicate poor cardiac fitness"
            )

        # Protective factors
        if age < 45:
            protective_factors.append("Young age is protective against heart disease")
        if trestbps < 120:
            protective_factors.append("Normal blood pressure is protective")
        if chol < 200:
            protective_factors.append("Normal cholesterol levels are protective")
        if thalach > 160:
            protective_factors.append("Good exercise capacity indicates healthy heart")

        # Build explanation
        explanation = f"""
🏥 CLINICAL INTERPRETATION

📊 Prediction: {'High Risk for Heart Disease' if prediction == 1 else 'Low Risk for Heart Disease'}
📈 Model Confidence: {max(probability) * 100:.1f}%

🔴 RISK FACTORS IDENTIFIED:
"""
        for factor in risk_factors:
            explanation += f"• {factor}\n"

        explanation += f"""
🟢 PROTECTIVE FACTORS:
"""
        for factor in protective_factors:
            explanation += f"• {factor}\n"

        explanation += f"""
💡 RECOMMENDATIONS:
{'• Immediate consultation with a cardiologist recommended' if prediction == 1 else '• Continue regular health maintenance'}
{'• Consider stress testing or cardiac imaging' if prediction == 1 else '• Annual cardiovascular screening sufficient'}
• Maintain healthy lifestyle: diet, exercise, no smoking
• Monitor blood pressure and cholesterol regularly
• Follow up with healthcare provider as recommended

⚠️ DISCLAIMER: This prediction is for educational purposes only and should not replace professional medical advice.
"""

        self.explanation_text.setText(explanation)

    def clear_fields(self):
        """Clear all input fields"""
        defaults = [
            "40",
            "1",
            "1",
            "120",
            "200",
            "0",
            "0",
            "150",
            "0",
            "1.0",
            "1",
        ]
        keys = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
        ]

        for key, default in zip(keys, defaults):
            self.input_fields[key].setText(default)

        self.prediction_label.setText("No prediction yet")
        self.prediction_label.setStyleSheet("")
        self.probability_label.setText("")
        self.confidence_label.setText("")
        self.explanation_text.clear()

    def load_sample_data(self):
        """Load sample patient data"""
        sample_data = [
            "63",
            "1",
            "3",
            "145",
            "233",
            "1",
            "0",
            "150",
            "0",
            "2.3",
            "0",
        ]
        keys = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
        ]

        for key, value in zip(keys, sample_data):
            self.input_fields[key].setText(value)

    def apply_styles(self):
        """Apply modern styling to the application"""
        style = """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 10px;
            background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
            background-color: #f5f5f5;
        }
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 12px;
            text-align: center;
            font-size: 14px;
            border-radius: 8px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        QLineEdit {
            border: 2px solid #ddd;
            border-radius: 4px;
            padding: 8px;
            font-size: 12px;
            background-color: white;
        }
        QLineEdit:focus {
            border-color: #4CAF50;
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: white;
        }
        QTabBar::tab {
            background-color: #e1e1e1;
            padding: 10px 15px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        QTabBar::tab:selected {
            background-color: #4CAF50;
            color: white;
        }
        """
        self.setStyleSheet(style)


def main():
    app = QApplication(sys.argv)
    window = HeartDiseaseApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
