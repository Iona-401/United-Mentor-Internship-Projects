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

class HeartDiseaseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Disease Prediction App")
        self.setGeometry(100, 100, 800, 600)
        self.setFont(QFont("Gothic", 12))
        self.model = joblib.load("Heart Disease/heart_disease_model.pkl")
        self.scaler = joblib.load("Heart Disease/heart_disease_scaler.pkl")
        self.dataset = pd.read_csv("Heart Disease/dataset.csv")
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
        
        title = QLabel("Heart Disease Prediction App")
        title.setFont(QFont("Gothic", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Form Fields
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(20)
        
        self.add_input_field("age", "Age")
        self.add_input_field("resting bp s", "Resting Blood Pressure (s)")
        self.add_input_field("cholesterol", "Cholesterol")
        self.add_input_field("max heart rate", "Max Heart Rate")
        self.add_input_field("oldpeak", "Oldpeak")
        
        self.add_combo_field("sex", "Sex", {
            "Male": 1, 
            "Female": 0
        })
        self.add_combo_field("chest pain type", "Chest Pain Type", {
            "Typical Angina": 0, 
            "Atypical Angina": 1, 
            "Non-Anginal Pain": 2, 
            "Asymptomatic": 3
        })
        self.add_combo_field("fasting blood sugar", "Fasting Blood Sugar > 120 mg/dl", {
            "Yes": 1, 
            "No": 0
        })
        self.add_combo_field("resting ecg", "Resting Electrocardiographic Results", {
            "Normal": 0, 
            "ST-T Wave Abnormality": 1, 
            "Left Ventricular Hypertrophy": 2
        })
        self.add_combo_field("exercise angina", "Exercise Induced Angina", {
            "Yes": 1, 
            "No": 0
        })
        self.add_combo_field("ST slope", "Slope of the Peak Exercise ST Segment", {
            "Upsloping": 0, 
            "Flat": 1, 
            "Downsloping": 2
        })

        form_layout.addRow(QLabel("Age:"), self.fields["age"])
        form_layout.addRow(QLabel("Sex:"), self.fields["sex"])
        form_layout.addRow(QLabel("Chest Pain Type:"), self.fields["chest pain type"])
        form_layout.addRow(QLabel("Resting BP:"), self.fields["resting bp s"])
        form_layout.addRow(QLabel("Cholesterol:"), self.fields["cholesterol"])
        form_layout.addRow(QLabel("Fasting Blood Sugar:"), self.fields["fasting blood sugar"])
        form_layout.addRow(QLabel("Resting ECG:"), self.fields["resting ecg"])
        form_layout.addRow(QLabel("Max Heart Rate:"), self.fields["max heart rate"])
        form_layout.addRow(QLabel("Exercise Angina:"), self.fields["exercise angina"])
        form_layout.addRow(QLabel("Oldpeak:"), self.fields["oldpeak"])
        form_layout.addRow(QLabel("ST Slope:"), self.fields["ST slope"])
        
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
        #results_section.addWidget(self.graph_title_label)

        self.canvas = None
        
        # Matplotlib Canvas Placeholder
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

            feature_names = ["age", "sex", "chest pain type", "resting bp s", "cholesterol", "fasting blood sugar", "resting ecg", "max heart rate", "exercise angina", "oldpeak", "ST slope"]

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
            
            feature_names = ["age", "sex", "chest pain type", "resting bp s", "cholesterol", "fasting blood sugar", "resting ecg", "max heart rate", "exercise angina", "oldpeak", "ST slope"]
            input_df = pd.DataFrame([input_data], columns=feature_names)
            scaled_input = self.scaler.transform(input_df)
            prediction = self.model.predict(scaled_input)[0]
            probability = self.model.predict_proba(scaled_input)[0][1]
            
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

    def plot_patient_graphs(self, input_df):
        if self.canvas:
            self.canvas.setParent(None)

        # Set up dark theme for matplotlib
        plt.close('all')
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), facecolor='#2b2b2b')
        fig.tight_layout(pad=4.0)
        fig.suptitle("Patient Data Visualization", fontsize=16, color="#FFF", weight="bold")

        features = ["age", "cholesterol", "resting bp s", "max heart rate"]
        plot_names = {
            "age": "Age",
            "resting bp s": "Resting Blood Pressure",
            "cholesterol": "Cholesterol",
            "max heart rate": "Max Heart Rate",
        }
        xlimits = {
            "age": (25, 80),
            "resting bp s": (80, 200),
            "cholesterol": (100, 400),
            "max heart rate": (60, 210),
        }

        # Set seaborn style for dark mode
        sns.set_theme(style="darkgrid", rc={
            "axes.facecolor": "#2b2b2b",
            "axes.edgecolor": "#888",
            "axes.labelcolor": "#fff",
            "xtick.color": "#fff",
            "ytick.color": "#fff",
            "grid.color": "#444",
            "text.color": "#fff",
            "figure.facecolor": "#2b2b2b",
            "legend.facecolor": "#444",
            "legend.edgecolor": "#888",
        })

        for ax, feature in zip(axs.flat, features):
            sns.histplot(self.dataset[feature], kde=True, ax=ax, color="#66b3ff", alpha=0.7)
            ax.axvline(input_df[feature].iloc[0], color="red", linestyle="--", label="Patient", linewidth=2)
            ax.set_xlim(*xlimits[feature])
            ax.set_title(plot_names[feature], color="#FFF", fontsize=12, weight="bold")
            ax.legend(facecolor="#444", edgecolor="#888", labelcolor="#FFF")
            ax.set_xlabel(feature, color="#FFF")
            ax.set_ylabel("Count", color="#FFF")
            # Set ticks color
            ax.tick_params(colors="#FFF")
            # Set spine color
            for spine in ax.spines.values():
                spine.set_color("#888")
            # Set grid color
            ax.grid(color="#444")

        self.canvas = FigureCanvas(fig)
        self.canvas_layout.addWidget(self.canvas)

    def get_x_limits(self, feature):
        limits = {
            "age": (25, 80),
            "resting bp s": (80, 200),
            "cholesterol": (100, 400),
            "max heart rate": (60, 210),
        }
        return limits.get(feature, (None, None))
    
    def get_plot_names(self, feature):
        names = {
            "age": "Age",
            "resting bp s": "Resting Blood Pressure",
            "cholesterol": "Cholesterol",
            "max heart rate": "Max Heart Rate",
        }
        return names.get(feature, feature.capitalize())
        
    def on_clear(self):
        for key in self.fields:
            if isinstance(self.fields[key], QLineEdit):
                self.fields[key].clear()
            elif isinstance(self.fields[key], QComboBox):
                self.fields[key].setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    HeartDiseaseApp.apply_dark_theme(app)
    window = HeartDiseaseApp()
    window.show()
    sys.exit(app.exec_())