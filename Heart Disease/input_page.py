import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QMessageBox, QWidget, QLineEdit, QFormLayout, QComboBox
from PyQt5.QtGui import QFont

import joblib
from output_page import OutputPage

class HeartDiseaseInputForm(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model = joblib.load("heart_disease_model.pkl")

    def init_ui(self):
        self.setWindowTitle('Heart Disease Input Form')
        self.setGeometry(100, 100, 400, 600)
        self.setFont(QFont('Gothic', 12))
        self.init_ui()
    
    def init_ui(self):
        layout = QFormLayout()
        layout.setVerticalSpacing(20)
        self.fields = {}
        
        #Text Boxes
        self.fields["age"] = QLineEdit()
        self.fields["restbp"] = QLineEdit()
        self.fields["chol"] = QLineEdit()
        self.fields["mhr"] = QLineEdit()
        self.fields["oldpeak"] = QLineEdit()

        #Dropdowms
        self.fields["sex"] = QComboBox()
        self.fields["cp"] = QComboBox()
        self.fields["fbs"] = QComboBox()
        self.fields["restecg"] = QComboBox()
        self.fields["exang"] = QComboBox()
        self.fields["slope"] = QComboBox()
        
        #Adding options to dropdowns
        self.fields["sex"].addItem("--Select--", None)
        self.fields["sex"].addItem("Female", 0)
        self.fields["sex"].addItem("Male", 1)
        
        self.fields["cp"].addItem("--Select--", None)
        self.fields["cp"].addItem("Typical Angina", 0)
        self.fields["cp"].addItem("Atypical Angina", 1)
        self.fields["cp"].addItem("Non-Anginal Pain", 2)
        self.fields["cp"].addItem("Asymptomatic", 3)
        
        self.fields["fbs"].addItem("--Select--", None)
        self.fields["fbs"].addItem("Blood Sugar < 120 mg/dl", 0)
        self.fields["fbs"].addItem("Blood Sugar >= 120 mg/dl", 1)
        
        self.fields["restecg"].addItem("--Select--", None)
        self.fields["restecg"].addItem("Normal", 0)
        self.fields["restecg"].addItem("ST-T Wave Abnormality", 1)
        self.fields["restecg"].addItem("Left Ventricular Hypertrophy", 2)
        
        self.fields["exang"].addItem("--Select--", None)
        self.fields["exang"].addItem("No", 0)
        self.fields["exang"].addItem("Yes", 1)
        
        self.fields["slope"].addItem("--Select--", None)
        self.fields["slope"].addItem("Upsloping", 0)
        self.fields["slope"].addItem("Flat", 1)
        self.fields["slope"].addItem("Downsloping", 2)

        #Layout for the form
        layout.addRow(QLabel('Heart Disease Input Form'), QLabel())
        layout.addRow("Age:", self.fields["age"])
        layout.addRow("Sex:", self.fields["sex"])
        layout.addRow("Chest Pain Type:", self.fields["cp"])
        layout.addRow("Resting Blood Pressure:", self.fields["restbp"])
        layout.addRow("Cholesterol:", self.fields["chol"])
        layout.addRow("Fasting Blood Sugar:", self.fields["fbs"])
        layout.addRow("Resting ECG:", self.fields["restecg"])
        layout.addRow("Maximum Heart Rate:", self.fields["mhr"])
        layout.addRow("Exercise Induced Angina:", self.fields["exang"])
        layout.addRow("Old Peak:", self.fields["oldpeak"])
        layout.addRow("Slope of ST Segment:", self.fields["slope"])
        
        #Submit Button
        self.submit_button = QPushButton('Submit')
        self.submit_button.setFont(QFont('Gothic', 14))
        self.submit_button.clicked.connect(self.on_submit)
        layout.addRow(self.submit_button)
        
        self.setLayout(layout)
        
    def on_submit(self):
        try:
            input_data = {
                "age": float(self.fields["age"].text()),
                "sex": self.fields["sex"].currentData(),
                "cp": self.fields["cp"].currentData(),
                "restbp": float(self.fields["restbp"].text()),
                "chol": float(self.fields["chol"].text()),
                "fbs": self.fields["fbs"].currentData(),
                "restecg": self.fields["restecg"].currentData(),
                "mhr": float(self.fields["mhr"].text()),
                "exang": self.fields["exang"].currentData(),
                "oldpeak": float(self.fields["oldpeak"].text()),
                "slope": self.fields["slope"].currentData()
            }
            
            print("Patient Input:", input_data)
            QMessageBox.information(self, 'Success', 'Input data submitted successfully!')
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter valid numeric values for all fields.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = HeartDiseaseInputForm()
    form.show()
    sys.exit(app.exec_())