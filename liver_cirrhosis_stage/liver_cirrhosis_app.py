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

FILE_PATH = "liver_cirrhosis_stage/liver_cirrhosis.csv"
MODEL_PATH = "liver_cirrhosis_stage/xgboost_liver_cirrhosis_model.pkl"

class LiverCirrhosisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liver Cirrhosis Stage Prediction App")
        self.setGeometry(100, 100, 800, 600)
        self.setFont(QFont("Gothic", 12))
        self.model = joblib.load(MODEL_PATH)
        self.dataset = pd.read_csv(FILE_PATH)
        self.fields = {}
        self.init_ui()

    def init_ui(self):
        top_layout = QVBoxLayout()
        secondary_layout = QHBoxLayout()
        form_layout = QVBoxLayout()
        plot_layout = QVBoxLayout()
        
        title = QLabel("Liver Cirrhosis Stage Prediction")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Gothic", 20, QFont.Bold))
        top_layout.addWidget(title)
        
        # Form fields
        self.add_input_field("N_Days", "Number of Days")
        self.add_input_field("Age", "Age")
        self.add_input_field("Bilirubin", "Total Bilirubin")
        self.add_input_field("Cholesterol", "Cholesterol")
        self.add_input_field("Albumin", "Albumin")
        self.add_input_field("Copper", "Copper")
        self.add_input_field("Alk_phos", "Alkaline Phosphatase")
        self.add_input_field("SGOT", "SGOT")
        self.add_input_field("Triglicerides", "Triglycerides Level")
        self.add_input_field("Platelets", "Platelet Count")
        self.add_input_field("Prothrombin", "Prothrombin Time")

        self.add_combo_field("Status", "Status", {
            "Censored": 0,
            "Censored due to Liver TX": 1,
            "Death": 2
        })
        self.add_combo_field("Drug", "Drug", {
            "D-Penicillamine": 1,
            "Placebo": 0
        })
        self.add_combo_field("Sex", "Gender", {
            "Male": 1, 
            "Female": 0
        })
        self.add_combo_field("Ascites", "Ascites", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("Hepatomegaly", "Hepatomegaly", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("Spiders", "Spiders", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("Edema", "Edema", {
            "Yes": 2,
            "Edema": 1,
            "No": 0
        })

        form_section = QFormLayout()
        form_section.setVerticalSpacing(20)

        form_section.addRow(QLabel("Number of Days:"), self.fields["N_Days"])
        form_section.addRow(QLabel("Age:"), self.fields["Age"])
        form_section.addRow(QLabel("Status:"), self.fields["Status"])
        form_section.addRow(QLabel("Drug:"), self.fields["Drug"])
        form_section.addRow(QLabel("Sex:"), self.fields["Sex"])
        form_section.addRow(QLabel("Ascites:"), self.fields["Ascites"])
        form_section.addRow(QLabel("Hepatomegaly:"), self.fields["Hepatomegaly"])
        form_section.addRow(QLabel("Spiders:"), self.fields["Spiders"])
        form_section.addRow(QLabel("Edema:"), self.fields["Edema"])
        form_section.addRow(QLabel("Total Bilirubin:"), self.fields["Bilirubin"])
        form_section.addRow(QLabel("Cholesterol:"), self.fields["Cholesterol"])
        form_section.addRow(QLabel("Albumin:"), self.fields["Albumin"])
        form_section.addRow(QLabel("Alkaline Phosphatase:"), self.fields["Alk_phos"])
        form_section.addRow(QLabel("SGOT:"), self.fields["SGOT"])
        form_section.addRow(QLabel("Triglycerides Level:"), self.fields["Triglicerides"])
        form_section.addRow(QLabel("Platelet Count:"), self.fields["Platelets"])
        form_section.addRow(QLabel("Prothrombin Time:"), self.fields["Prothrombin"])

        form_layout.addLayout(form_section)
        
        self.submit_button = QPushButton("Predict")
        self.submit_button.setFont(QFont("Gothic", 14))
        self.submit_button.clicked.connect(self.make_prediction)
        self.submit_button.setToolTip("Submit the form for prediction")
        form_layout.addWidget(self.submit_button)
        
        # Clear Form Button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setFont(QFont("Gothic", 14))
        self.clear_button.clicked.connect(self.on_clear)
        self.clear_button.setToolTip("Clear the form fields")
        form_layout.addWidget(self.clear_button)

        secondary_layout.addLayout(form_layout)
        top_layout.addLayout(secondary_layout)
        self.setLayout(top_layout)

    def add_input_field(self, key, label):
        field = QLineEdit()
        field.setPlaceholderText(label)
        self.fields[key] = field

    def add_combo_field(self, key, label, options):
        field = QComboBox()
        field.addItems(options)
        self.fields[key] = field
        
    def make_prediction(self):
        try:

            feature_names = ["N_Days", "Status", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]

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

            feature_names = ["N_Days", "Status", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]
            input_df = pd.DataFrame([input_data], columns=feature_names)
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0][1]

            if prediction == 1:
                self.result_label.setText(f"❗ Heart Disease Detected\nProbability: {probability:.2f}")
                self.result_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.result_label.setText(f"✅ No Heart Disease Detected\nProbability: {probability:.2f}")
                self.result_label.setStyleSheet("color: green; font-weight: bold;")
            
            self.plot_patient_graphs(input_df)
        
        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction Failed: {e}")