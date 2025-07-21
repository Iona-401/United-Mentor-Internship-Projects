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