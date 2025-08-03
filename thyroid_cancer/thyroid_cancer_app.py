import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QComboBox, QPushButton, QMessageBox, QFrame
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import joblib
import pandas as pd

# Fix the model path for both development and executable
def get_resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

MODEL_PATH = get_resource_path("random_forest_thyroid_cancer_model.pkl")
feature_names = ["Age", "Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy", "Thyroid Function", "Physical Examination", "Adenopathy", "Pathology", "Focality", "Risk", "T", "N", "M", "Stage", "Response"]

class ThyroidCancerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thyroid Cancer Prediction App")
        self.setGeometry(100, 100, 800, 600)
        self.setFont(QFont("Gothic", 12))
        self.model = joblib.load(MODEL_PATH)
        self.fields = {}
        self.init_ui()

    def init_ui(self):
        
        top_layout = QVBoxLayout()
        second_layout = QHBoxLayout()
        form_layout = QVBoxLayout()
        form_section = QFormLayout()
        result_layout = QVBoxLayout()
        title = QLabel("Thyroid Cancer Prediction")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Gothic", 20, QFont.Bold))
        top_layout.addWidget(title)        
        
        # Form fields
        self.add_input_field("Age", "Age")
        
        self.add_combo_field("Gender", "Gender", {
            "Male": 1,
            "Female": 0
        })
        self.add_combo_field("Smoking", "Smoking", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("Hx Smoking", "Hx Smoking", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("Hx Radiothreapy", "Hx Radiothreapy", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("Thyroid Function", "Thyroid Function", {
            "Euthyroid": 4,
            "Clinical Hyperthyroidism": 3,
            "Clinical Hypothyroidism": 2,
            "Subclinical Hyperthyroidism": 1,
            "Subclinical Hypothyroidism": 0
        })
        self.add_combo_field("Physical Examination", "Physical Examination", {
            "Multinodular goiter": 4,
            "Single modular goiter-left": 3,
            "Single modular goiter-right": 2,
            "Diffuse goiter": 1,
            "Normal": 0
        })
        self.add_combo_field("Adenopathy", "Adenopathy", {
            "Extensive": 5,
            "Bilateral": 4,
            "Posterior": 3,
            "Left": 2,
            "Right": 1,
            "No": 0
        })
        self.add_combo_field("Pathology", "Pathology", {
            "Papillary": 3,
            "Follicular": 2,
            "Medullary": 1,
            "Hurthle Cell": 0
        })
        self.add_combo_field("Focality", "Focality", {
            "Unifocal": 1,
            "Multifocal": 0
        })
        self.add_combo_field("Risk", "Risk", {
            "High": 2,
            "Intermediate": 1,
            "Low": 0
        })
        self.add_combo_field("T", "T", {
            "T4b": 6,
            "T4a": 5,
            "T3b": 4,
            "T3a": 3,
            "T2": 2,
            "T1b": 1,
            "T1a": 0
        })
        self.add_combo_field("N", "N", {
            "N1b": 3,
            "N1a": 2,
            "Nx": 1,
            "N0": 0
        })
        self.add_combo_field("M", "M", {
            "M1": 1,
            "M0": 0
        })
        self.add_combo_field("Stage", "Stage", {
            "IVB": 4,
            "IVA": 3,
            "III": 2,
            "II": 1,
            "I": 0
        })
        self.add_combo_field("Response", "Response", {
            "Excellent": 3,
            "Biochemical Incomplete": 2,
            "Structural Incomplete": 1,
            "Intermediate": 0
        })

        # Form Layout
        form_section.addRow(QLabel("Age:"), self.fields["Age"])
        form_section.addRow(QLabel("Gender:"), self.fields["Gender"])
        form_section.addRow(QLabel("Smoking:"), self.fields["Smoking"])
        form_section.addRow(QLabel("Hx Smoking:"), self.fields["Hx Smoking"])
        form_section.addRow(QLabel("Hx Radiothreapy:"), self.fields["Hx Radiothreapy"])
        form_section.addRow(QLabel("Thyroid Function:"), self.fields["Thyroid Function"])
        form_section.addRow(QLabel("Physical Examination:"), self.fields["Physical Examination"])
        form_section.addRow(QLabel("Adenopathy:"), self.fields["Adenopathy"])
        form_section.addRow(QLabel("Pathology:"), self.fields["Pathology"])
        form_section.addRow(QLabel("Focality:"), self.fields["Focality"])
        form_section.addRow(QLabel("Cancer Risk Category:"), self.fields["Risk"])
        form_section.addRow(QLabel("Tumor Classification (Size):"), self.fields["T"])
        form_section.addRow(QLabel("Nodal Classification:"), self.fields["N"])
        form_section.addRow(QLabel("Metastasis:"), self.fields["M"])
        form_section.addRow(QLabel("Cancer Stage:"), self.fields["Stage"])
        form_section.addRow(QLabel("Treatment Response:"), self.fields["Response"])

        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)

        # Submit Button
        self.submit_button = QPushButton("Predict Recurrence")
        self.submit_button.setFont(QFont("Gothic", 14))
        self.submit_button.clicked.connect(self.make_prediction)
        self.submit_button.setToolTip("Submit the form for prediction")

        # Clear Form Button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setFont(QFont("Gothic", 14))
        self.clear_button.clicked.connect(self.on_clear)
        self.clear_button.setToolTip("Clear the form fields")
        
        form_widget = QWidget()
        form_widget.setFixedWidth(400)
        
        form_layout.addLayout(form_section)
        form_layout.addWidget(form_widget)
        form_layout.addStretch(20)
        form_layout.addWidget(h_line)
        form_layout.addWidget(self.submit_button)
        form_layout.addWidget(self.clear_button)
        
        # Result Label
        result_widget = QWidget()
        result_widget.setFixedWidth(400)
        result_layout.addWidget(result_widget)

        self.result_title = QLabel("Prediction Result:")
        self.result_label = QLabel("")
        self.result_label.setObjectName("result_label")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_title.setAlignment(Qt.AlignCenter)
        
        result_layout.addStretch(1)
        result_layout.addWidget(self.result_title)
        result_layout.addWidget(self.result_label)
        result_layout.addStretch(1)
        
        v_line = QFrame()
        v_line.setFrameShape(QFrame.VLine)
        v_line.setFrameShadow(QFrame.Sunken)
        v_line.setFixedHeight(400)
        
        second_layout.addLayout(form_layout)
        second_layout.addWidget(v_line)
        second_layout.addLayout(result_layout)
        top_layout.addLayout(second_layout)
        self.setLayout(top_layout)

    def add_input_field(self, key, label):
        field = QLineEdit()
        field.setPlaceholderText(label)
        self.fields[key] = field

    def add_combo_field(self, key, label, options: dict):
        combo = QComboBox()
        combo.addItem("-- Select --", None)
        for name, value in options.items():
            combo.addItem(name, value)
        self.fields[key] = combo
        
    def make_prediction(self):
        try:

            feature_names = ["Age", "Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy", "Thyroid Function", "Physical Examination", "Adenopathy", "Pathology", "Focality", "Risk", "T", "N", "M", "Stage", "Response"]

            input_data = []
            
            for key in feature_names:
                widget = self.fields[key]
                if isinstance(widget, QLineEdit):
                    value = widget.text()
                    if not value:
                        raise ValueError(f"{key.capitalize()} cannot be empty.")
                    input_data.append(float(value))
                elif isinstance(widget, QComboBox):
                    value = widget.currentData()
                    if value is None:
                        raise ValueError(f"Please select a valid option for {key}.")
                    input_data.append(value)

            input_df = pd.DataFrame([input_data], columns=feature_names)
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0][1]

            if prediction == 1:
                self.result_label.setText(f"Cancer Has Recurred\nProbability: {probability:.2f}")
                self.result_label.setStyleSheet("color: red; font-weight: bold;")
            elif prediction == 0:
                self.result_label.setText(f"Cancer Has Not Recurred\nProbability: {probability:.2f}")
                self.result_label.setStyleSheet("color: green; font-weight: bold;")
        
        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction Failed: {e}")
        
    def on_clear(self):
        for key in self.fields:
            if isinstance(self.fields[key], QLineEdit):
                self.fields[key].clear()
            elif isinstance(self.fields[key], QComboBox):
                self.fields[key].setCurrentIndex(0)
    
    @staticmethod
    def apply_dark_theme(app):
        dark_style = """
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: Gothic;
                font-size: 12pt;
            }

            QLineEdit, QComboBox {
                background-color: #3c3f41;
                border: 1px solid #555;
                color: #fff;
                padding: 5px;
                border-radius: 4px;
            }

            QPushButton {
                background-color: #4e8ef7;
                color: white;
                padding: 6px 12px;
                border-radius: 5px;
            }

            QPushButton:hover {
                background-color: #6faaff;
            }

            QLabel#result_label {
                font-size: 14pt;
                font-weight: bold;
            }
        """
        app.setStyleSheet(dark_style)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ThyroidCancerApp.apply_dark_theme(app)
    window = ThyroidCancerApp()
    window.show()
    sys.exit(app.exec_())