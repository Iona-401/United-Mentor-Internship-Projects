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

class LiverCirrhosisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liver Cirrhosis Stage Prediction App")
        self.setGeometry(100, 100, 800, 600)
        self.setFont(QFont("Gothic", 12))
        
        # Get the directory of the executable or script
        if getattr(sys, 'frozen', False):
            # Running as compiled executable - use sys._MEIPASS for bundled files
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        else:
            # Running as script
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load model and data files
        try:
            self.model = joblib.load(os.path.join(base_dir, "random_forest_liver_cirrhosis_model.pkl"))
            self.dataset = pd.read_csv(os.path.join(base_dir, "liver_cirrhosis.csv"))
        except FileNotFoundError as e:
            QMessageBox.critical(None, "File Error", f"Required file not found: {e}\nBase directory: {base_dir}")
            sys.exit(1)
            
        self.fields = {}
        self.init_ui()

    def init_ui(self):
        
        top_layout = QVBoxLayout()
        second_layout = QHBoxLayout()
        form_layout = QVBoxLayout()
        form_section = QFormLayout()
        result_layout = QVBoxLayout()
        
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
        self.add_input_field("Alk_Phos", "Alkaline Phosphatase")
        self.add_input_field("SGOT", "SGOT")
        self.add_input_field("Tryglicerides", "Triglycerides Level")
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
            "Edema Present": 2,
            "Edema not present with Diuretics": 1,
            "No Edema": 0
        })

        # Form Layout
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
        form_section.addRow(QLabel("Copper:"), self.fields["Copper"])
        form_section.addRow(QLabel("Alkaline Phosphatase:"), self.fields["Alk_Phos"])
        form_section.addRow(QLabel("SGOT:"), self.fields["SGOT"])
        form_section.addRow(QLabel("Triglycerides Level:"), self.fields["Tryglicerides"])
        form_section.addRow(QLabel("Platelet Count:"), self.fields["Platelets"])
        form_section.addRow(QLabel("Prothrombin Time:"), self.fields["Prothrombin"])
        
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)

        # Submit Button
        self.submit_button = QPushButton("Predict Stage")
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

            if prediction == 2:
                self.result_label.setText(f"Liver Cirrhosis Stage 3\nProbability: {probability:.2f}")
                self.result_label.setStyleSheet("color: red; font-weight: bold;")
            elif prediction == 1:
                self.result_label.setText(f"Liver Cirrhosis Stage 2\nProbability: {probability:.2f}")
                self.result_label.setStyleSheet("color: orange; font-weight: bold;")
            elif prediction == 0:
                self.result_label.setText(f"Liver Cirrhosis Stage 1\nProbability: {probability:.2f}")
                self.result_label.setStyleSheet("color: yellow; font-weight: bold;")
        
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
    LiverCirrhosisApp.apply_dark_theme(app)
    window = LiverCirrhosisApp()
    window.show()
    sys.exit(app.exec_())