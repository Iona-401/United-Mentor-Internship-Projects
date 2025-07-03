from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt

class OutputPage(QWidget):
    def __init__(self, prediction, probability = None):
        super().__init__()
        self.setWindowTitle('Heart Disease Prediction Result')
        self.setGeometry(100, 100, 400, 200)
        self.setFont(QFont('Gothic', 12))
        self.init_ui(prediction, probability)

    def init_ui(self, prediction, probability):
        layout = QVBoxLayout()
        
        label = QLabel()
        label.setFont(QFont('Gothic', 16))
        label.setAlignment(Qt.AlignCenter)

        if prediction == 1:
            label.setText("Heart Disease Detected")
            label.setStyleSheet("color: red;")
        else:
            label.setText("No Heart Disease Detected")
            label.setStyleSheet("color: green;")
        
        layout.addWidget(label)

        if probability is not None:
            prob_label = QLabel(f"Probability of Heart Disease: {probability*100}%")
            prob_label.setFont(QFont('Gothic', 14))
            prob_label.setAlignment(Qt.AlignCenter)
            prob_label.setStyleSheet("color: blue;")
            layout.addWidget(prob_label)
        
        self.setLayout(layout)