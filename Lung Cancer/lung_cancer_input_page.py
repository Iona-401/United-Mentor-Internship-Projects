import sys
from PyQt5.QtWidgets import (
    QWidget, QLabel, QApplication, QFormLayout, QLineEdit,
    QPushButton, QMessageBox, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import joblib
import pandas as pd
from lung_cancer_output_page import LungCancerOutputPage

class LungCancerInputForm(QWidget):
    def __init__(self):
        super().__init__()
        self.model = joblib.load("Lung Cancer/lung_cancer_rf_model.pkl")
        self.setWindowTitle("Lung Cancer Prediction")
        self.init_ui()
    
    def init_ui(self):
        layout = QFormLayout()
        layout.setVerticalSpacing(20)
        self.setGeometry(100, 100, 400, 600)
        self.setFont(QFont('Gothic', 12))
        self.fields = {}
        
        # Numerical Fields
        self.fields["age"] = QLineEdit()
        self.fields["bmi"] = QLineEdit()
        self.fields["cholesterol_level"] = QLineEdit()

        # Categorical Fields
        self.fields["gender"] = QComboBox()
        self.fields["cancer_stage"] = QComboBox()
        self.fields["smoking_status"] = QComboBox()
        self.fields["family_history"] = QCheckBox()
        self.fields["treatment_type"] = QComboBox()
        self.fields["hypertension"] = QCheckBox()
        self.fields["asthma"] = QCheckBox()
        self.fields["cirrhosis"] = QCheckBox()
        self.fields["other_cancer"] = QCheckBox()
        
        # Adding options to dropdowns
        self.fields["gender"].addItem("--Select--", None)
        self.fields["gender"].addItem("Male", 0)
        self.fields["gender"].addItem("Female", 1)
        
        self.fields["cancer_stage"].addItem("--Select--", None)
        self.fields["cancer_stage"].addItem("Stage I", 0)
        self.fields["cancer_stage"].addItem("Stage II", 1)
        self.fields["cancer_stage"].addItem("Stage III", 2)
        self.fields["cancer_stage"].addItem("Stage IV", 3)
        
        self.fields["smoking_status"].addItem("--Select--", None)
        self.fields["smoking_status"].addItem("Never Smoked", 0)
        self.fields["smoking_status"].addItem("Former Smoker", 1)
        self.fields["smoking_status"].addItem("Current Smoker", 2)
        
        self.fields["treatment_type"].addItem("--Select--", None)
        self.fields["treatment_type"].addItem("Surgery", 0)
        self.fields["treatment_type"].addItem("Chemotherapy", 1)
        self.fields["treatment_type"].addItem("Radiation", 2)
        self.fields["treatment_type"].addItem("Immunotherapy", 3)
        
        # Checkboxes
        self.fields["family_history"].setText("Family History")
        self.fields["hypertension"].setText("Hypertension")
        self.fields["asthma"].setText("Asthma")
        self.fields["cirrhosis"].setText("Cirrhosis")
        self.fields["other_cancer"].setText("Other Cancer")
        
        # Adding fields to layout
        layout.addRow(QLabel("Age:"), self.fields["age"])
        layout.addRow(QLabel("BMI:"), self.fields["bmi"])
        layout.addRow(QLabel("Cholesterol:"), self.fields["cholesterol_level"])
        layout.addRow(QLabel("Gender:"), self.fields["gender"])
        layout.addRow(QLabel("Cancer Stage:"), self.fields["cancer_stage"])
        layout.addRow(QLabel("Smoking Status:"), self.fields["smoking_status"])
        layout.addRow(QLabel("Family History:"), self.fields["family_history"])
        layout.addRow(QLabel("Treatment Type:"), self.fields["treatment_type"])
        layout.addRow(QLabel("Hypertension:"), self.fields["hypertension"])
        layout.addRow(QLabel("Asthma:"), self.fields["asthma"])
        layout.addRow(QLabel("Cirrhosis:"), self.fields["cirrhosis"])
        layout.addRow(QLabel("Other Cancer:"), self.fields["other_cancer"])

        self.submit_button = QPushButton('Submit')
        self.submit_button.setFont(QFont('Gothic', 14))
        self.submit_button.clicked.connect(self.on_submit)
        layout.addRow(self.submit_button)
        
        self.setLayout(layout)
    
    def on_submit(self):
        try:
            input_data = []
            for key in ["age", "bmi", "cholesterol_level"]:
                text = self.fields[key].text()
                if text.strip() == "":
                    raise ValueError(f"Please enter a value for {key}")
                value = float(text)
                input_data.append(value)

            for key in ["gender", "cancer_stage", "smoking_status", "treatment_type"]:
                value = self.fields[key].currentData()
                if value is None:
                    raise ValueError(f"Please select a value for {key}")
                input_data.append(value)

            for key in ["family_history", "hypertension", "asthma", "cirrhosis", "other_cancer"]:
                value = self.fields[key].isChecked()
                input_data.append(1 if value else 0)
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")
            return  # Stop further execution if input is invalid

        # Prediction
        input_df = pd.DataFrame([input_data], columns=[
            "age", "bmi", "cholesterol_level", "gender", "cancer_stage", "smoking_status", 
            "treatment_type", "family_history", "hypertension", "asthma", "cirrhosis", "other_cancer" 
        ])
        prediction = self.model.predict(input_df)
        try:
            prediction_prob = self.model.predict_proba(input_df)[0][1]
        except Exception:
            prediction_prob = None

        self.result_page = LungCancerOutputPage(prediction, prediction_prob, input_data)
        self.result_page.show()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = LungCancerInputForm()
    form.show()
    sys.exit(app.exec_())