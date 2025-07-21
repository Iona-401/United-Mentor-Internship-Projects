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
        form_layout = QFormLayout()
        plot_layout = QVBoxLayout()
        
        title = QLabel("Liver Cirrhosis Stage Prediction")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Gothic", 20, QFont.Bold))
        top_layout.addWidget(title)
        
        # Form fields
        self.add_input_field("N_Days", "Number of Days")
        self.add_input_field("Age", "Age")
        self.add_input_field("Bilirubin", "Total Bilirubin")
        self.add_input_field("Alk_phos", "Alkaline Phosphatase")
        self.add_input_field("SGOT", "SGOT")
        self.add_input_field("Albumin", "Albumin")
        self.add_input_field("Protime", "Prothrombin Time")

        self.add_combo_field("gender", "Gender", {
            "Male": 1, 
            "Female": 0
        })
        self.add_combo_field("status", "Status", {
            "Censored": 0,
            "Censored ": 1,
            "D": 2
        })
        self.add_combo_field("drug", "Drug", {
            "D-Penicillamine": 1,
            "Placebo": 0
        })
        self.add_combo_field("ascites", "Ascites", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("hepatomegaly", "Hepatomegaly", {
            "Yes": 1,
            "No": 0
        })
        self.add_combo_field("spiders", "Spiders", {
            "Yes": 1,
            "No": 0
        })
        

        form_layout.addRow(QLabel("Age:"), self.fields["age"])
        form_layout.addRow(QLabel("Status:"), self.fields["status"])
        
        form_layout.addRow(QLabel("Total Bilirubin:"), self.fields["bilirubin"])
        form_layout.addRow(QLabel("Alkaline Phosphatase:"), self.fields["alk_phosphate"])
        form_layout.addRow(QLabel("SGPT:"), self.fields["sgpt"])
        form_layout.addRow(QLabel("SGOT:"), self.fields["sgot"])
        form_layout.addRow(QLabel("Albumin:"), self.fields["albumin"])
        form_layout.addRow(QLabel("Prothrombin Time:"), self.fields["protime"])

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