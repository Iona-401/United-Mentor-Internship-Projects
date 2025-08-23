import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QComboBox, QPushButton, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

class LungCancerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Cancer Prediction App")
        self.setGeometry(100, 100, 800, 600)
        self.setFont(QFont("Gothic", 12))
        self.model = joblib.load("Lung Cancer/lung_cancer_model.pkl")
        self.preprocessor = joblib.load("Lung Cancer/lung_cancer_preprocessor.pkl")
        self.dataset = pd.read_csv("Lung Cancer/dataset_med.csv")
        self.fields = {}
        self.init_ui()
    
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
    
    def init_ui(self):
        # Form Layout
        main_layout = QVBoxLayout()
        second_layout = QHBoxLayout()
        form_section = QVBoxLayout()
        results_section = QVBoxLayout()

        title = QLabel("Lung Cancer Prediction App")
        title.setFont(QFont("Gothic", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Form Fields
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(20)
        
        self.add_input_field("Age", "age")
        self.add_input_field("BMI", "bmi")
        self.add_input_field("Cholesterol Level", "cholesterol_level")

        self.add_combo_field("Gender", "gender", {
            "Male":0,
            "Female":1
        })
        self.add_combo_field("Cancer Stage", "cancer_stage", {
            "Stage I":0,
            "Stage II":1,
            "Stage III":2,
            "Stage IV":3
        })
        self.add_combo_field("Smoking Status", "smoking_status", {
            "Never Smoked":0,
            "Former Smoker":1,
            "Current Smoker":2
        })
        self.add_combo_field("Treatment Type", "treatment_type", {
            "Surgery":0,
            "Chemotherapy":1,
            "Radiation":2,
            "Immunotherapy":3
        })
        self.add_combo_field("Family History", "family_history", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("Hypertension", "hypertension", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("Asthma", "asthma", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("Cirrhosis", "cirrhosis", {
            "Yes": 1,
            "No": 0
        })
        
        form_layout.addRow(QLabel("Age:"), self.fields["age"])
        form_layout.addRow(QLabel("BMI:"), self.fields["bmi"])
        form_layout.addRow(QLabel("Cholesterol:"), self.fields["cholesterol_level"])
        form_layout.addRow(QLabel("Gender:"), self.fields["gender"])
        form_layout.addRow(QLabel("Cancer Stage:"), self.fields["cancer_stage"])
        form_layout.addRow(QLabel("Smoking Status:"), self.fields["smoking_status"])
        form_layout.addRow(QLabel("Family History:"), self.fields["family_history"])
        form_layout.addRow(QLabel("Treatment Type:"), self.fields["treatment_type"])
        form_layout.addRow(QLabel("Hypertension:"), self.fields["hypertension"])
        form_layout.addRow(QLabel("Asthma:"), self.fields["asthma"])
        form_layout.addRow(QLabel("Cirrhosis:"), self.fields["cirrhosis"])
        form_layout.addRow(QLabel("Other Cancer:"), self.fields["other_cancer"])

        form_section.addLayout(form_layout)
        
        # Predict Button
        self.submit_button = QPushButton("Predict")
        self.submit_button.setFont(QFont("Gothic", 14))
        self.submit_button.clicked.connect(self.make_prediction)
        self.submit_button.setToolTip("Submit the form for prediction")
        form_section.addWidget(self.submit_button)

        # Clear Form Button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setFont(QFont("Gothic", 14))
        self.clear_button.clicked.connect(self.on_clear)
        self.clear_button.setToolTip("Clear the form fields")
        form_section.addWidget(self.clear_button)
        
        form_section.addStretch()

        # Result Label
        self.result_label = QLabel("Prediction Result will be displayed here")
        self.result_label.setFont(QFont("Gothic", 14))
        self.result_label.setAlignment(Qt.AlignCenter)
        results_section.addWidget(self.result_label)
        
        # Matplotlib Canvas Placeholder
        self.canvas = None
        self.canvas_layout = None
        self.canvas_layout = QVBoxLayout()
        results_section.addLayout(self.canvas_layout)
        results_section.addStretch()

        second_layout.addLayout(form_section, 1)
        second_layout.addLayout(results_section, 2)
        main_layout.addLayout(second_layout)
        self.setLayout(main_layout)

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

            feature_names = ["age", "bmi", "cholesterol_level", "gender", "cancer_stage", "smoking_status", "family_history", "treatment_type", "hypertension", "asthma", "cirrhosis", "other_cancer"]
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

            feature_names = ["age", "bmi", "cholesterol_level", "gender", "cancer_stage", "smoking_status", "family_history", "treatment_type", "hypertension", "asthma", "cirrhosis", "other_cancer"]
            input_df = pd.DataFrame([input_data], columns=feature_names)
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0][1]
            print(f"Prediction: {prediction}, Probability: {probability:.2f}")
            if prediction == 1:
                self.result_label.setText(f"❗ Patient May Not Survive\nProbability: {probability:.2f}")
                self.result_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.result_label.setText(f"✅ Patient has a good chance of survival\nProbability: {probability:.2f}")
                self.result_label.setStyleSheet("color: green; font-weight: bold;")
            
            self.plot_patient_graphs(input_df)
        
        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction Failed: {e}")