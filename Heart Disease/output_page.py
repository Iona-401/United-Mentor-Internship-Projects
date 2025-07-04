import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class OutputPage(QWidget):
    def __init__(self, prediction, probability, input_data, dataset_path="Heart Disease/dataset.csv"):
        super().__init__()
        self.setWindowTitle('Heart Disease Prediction Result')
        self.setGeometry(100, 100, 400, 200)
        self.setFont(QFont('Gothic', 12))
        
        self.input_data = input_data
        self.dataset = pd.read_csv(dataset_path)
        self.prediction = prediction
        self.probability = probability

        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        label = QLabel()
        label.setFont(QFont('Gothic', 16))
        label.setAlignment(Qt.AlignCenter)

        if self.prediction == 1:
            label.setText("Heart Disease Detected")
            label.setStyleSheet("color: red;")
        else:
            label.setText("No Heart Disease Detected")
            label.setStyleSheet("color: green;")
        
        layout.addWidget(label)

        if self.probability is not None:
            prob_label = QLabel(f"Probability of Heart Disease: {self.probability:.2f}")
            prob_label.setAlignment(Qt.AlignCenter)
            prob_label.setFont(QFont('Gothic', 12))
            prob_label.setStyleSheet("color: blue;")
            layout.addWidget(prob_label)

        # Add graph canvas
        canvas = self.plot_patient_position()
        layout.addWidget(canvas)

        self.setLayout(layout)
        
    def plot_patient_position(self):
        """Plot where the patient lands in some key features"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.tight_layout(pad=3.0)
        fig.suptitle("Patient Position vs Dataset", fontsize=14)

        # Use correct dataset feature names
        feature_names = ["age", "cholesterol", "max heart rate", "oldpeak"]
        patient_values = {
            "age": self.input_data[0],
            "cholesterol": self.input_data[4],
            "max heart rate": self.input_data[7],
            "oldpeak": self.input_data[9]
        }

        for ax, feature in zip(axes.flat, feature_names):
            sns.histplot(self.dataset[feature], kde=True, ax=ax, color='lightblue')
            ax.axvline(patient_values[feature], color='red', linestyle='--', label="Patient")
            ax.set_title(f"{feature.capitalize()} Distribution")
            ax.legend()

        canvas = FigureCanvas(fig)
        return canvas