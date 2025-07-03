import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QMessageBox, QWidget, QLineEdit, QFormLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class HeartDiseaseInputForm(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Heart Disease Input Form')
        self.setGeometry(100, 100, 400, 600)
        self.init_ui()
    
    def init_ui(self):
        layout = QFormLayout()
        layout.setVerticalSpacing(20)
        
        self.fields = {
            "Age": QLineEdit(),
            "Sex (0 = Female, 1 = Male)": QLineEdit(),
            "Chest Pain Type (0 - 3)": QLineEdit(),
            "Resting Blood Pressure": QLineEdit(),
            "Serum Cholestoral": QLineEdit(),
            "Fasting Blood Sugar (0 = No, 1 = Yes)": QLineEdit(),
            "Resting ECG": QLineEdit(),
            "Max Heart Rate": QLineEdit(),
            "Exercise Angina": QLineEdit(),
            "Oldpeak": QLineEdit(),
            "Slope": QLineEdit(),
            "Ca": QLineEdit(),
            "Thal": QLineEdit()
        }

        for label, field in self.fields.items():
            field.setPlaceholderText(label)
            layout.addRow(QLabel(label), field)
        
        self.submit_button = QPushButton('Submit')
        self.submit_button.setFont(QFont('Arial', 14))
        self.submit_button.clicked.connect(self.on_submit)
        
        layout.addRow(self.submit_button)
        
        self.setLayout(layout)
    
    def on_submit(self):
        try:
            input_data = {label: float(field.text()) for label, field in self.fields.items()}
            print("Input Data:", input_data)  # Replace with actual model prediction logic
            
            QMessageBox.information(self, 'Success', 'Data submitted successfully!')
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter valid numeric values for all fields.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = HeartDiseaseInputForm()
    form.show()
    sys.exit(app.exec_())