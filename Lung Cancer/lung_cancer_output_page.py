from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class LungCancerOutputPage(QWidget):
    def __init__(self, prediction, probability, input_data, dataset_path="Lung Cancer/dataset_med.csv"):
        super().__init__()
        self.setWindowTitle('Lung Cancer Prediction Result')
        self.setGeometry(100, 100, 400, 200)
        self.setFont(QFont('Gothic', 12))
        
        self.input_data = input_data
        self.dataset = pd.read_csv(dataset_path)
        self.prediction = prediction
        self.probability = probability

        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setFont(QFont('Gothic', 12))
        
        # Display prediction result
        prob_label = QLabel(f"Probability of Lung Cancer: {self.probability:.2f}")
        prob_label.setAlignment(Qt.AlignCenter)
        prob_label.setFont(QFont('Gothic', 12))
        prob_label.setStyleSheet("color: blue;")
        layout.addWidget(prob_label)
        
        self.setLayout(layout)
        