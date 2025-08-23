import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QFrame,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt5.QtGui import QPixmap, QFont, QDragEnterEvent, QDropEvent
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

MODEL_PATH = "Animal Classification/aniClass_EFF_Stage2.pkl"


class PredictionThread(QThread):
    prediction_ready = pyqtSignal(str, float)

    def __init__(self, image_path, model):
        super().__init__()
        self.image_path = image_path
        self.model = model

    def run(self):
        try:
            img = Image.open(self.image_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Set model to inference mode (disable augmentation)
            prediction = self.model(img_array, training=False)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class_idx]

            animal_classes = [
                "Bear",
                "Bird",
                "Cat",
                "Cow",
                "Deer",
                "Dog",
                "Dolphin",
                "Elephant",
                "Giraffe",
                "Horse",
                "Kangaroo",
                "Lion",
                "Panda",
                "Tiger",
                "Zebra",
            ]

            predicted_animal = animal_classes[predicted_class_idx]
            self.prediction_ready.emit(predicted_animal, confidence)

        except Exception as e:
            self.prediction_ready.emit(f"Error: {str(e)}", 0.0)


class DropArea(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            """
            QLabel{
            border: 3px dashed #999;
            border-radius: 10px;
            background-color: #f0f0f0;
            color: #666;
            font-size: 16px;
            font-family: "Gothic";
            padding: 40px;
        }
        QLabel:hover {
            background-color: #e0e0e0;
            border-color: #666;
        }
        """
        )
        self.setText("Drag & Drop an Image here\nor\nClick to Browse")
        self.setMinimumSize(300, 200)
        self.setFont(QFont("Gothic", 14))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(
                """
                QLabel {
                    border: 3px dashed #4CAF50;
                    border-radius: 10px;
                    background-color: #e8f5e8;
                    color: #2e7d32;
                    font-size: 16px;
                    font-family: "Gothic";
                    padding: 40px;
                }
            """
            )
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            """
        QLabel {
            border: 3px dashed #999;
            border-radius: 10px;
            background-color: #f0f0f0;
            color: #666;
            font-size: 16px;
            font-family: "Gothic";
            padding: 40px;
        }
        """
        )

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            file_path = files[0]
            if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                self.file_dropped.emit(file_path)

        self.setStyleSheet(
            """
                QLabel {
                    border: 3px dashed #999;
                    border-radius: 10px;
                    background-color: #f0f0f0;
                    color: #666;
                    font-size: 16px;
                    font-family: "Gothic";
                    padding: 40px;
                }
            """
        )

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.file_dropped.emit(file_path)


class AnimalClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.prediction_thread = None
        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle("Animal Classification App")
        self.setGeometry(100, 100, 800, 600)
        self.setFont(QFont("Gothic", 16))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        title_label = QLabel("Animal Classification")
        title_label.setFont(QFont("Gothic", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        content_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self.handle_image_drop)
        left_layout.addWidget(self.drop_area)

        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(224, 224)
        self.image_preview.setMaximumSize(224, 224)
        self.image_preview.setStyleSheet(
            """
                QLabel {
                    border: 2px solid #999;
                    border-radius: 10px;
                    background-color: #f0f0f0;
                    color: white;
                }
            """
        )
        self.image_preview.hide()
        left_layout.addWidget(self.image_preview)

        content_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()

        results_frame = QFrame()
        results_frame.setStyleSheet(
            """
            QFrame {
                border: 2px solid #999;
                border-radius: 10px;
                background-color: white;
                padding: 20px;
            }
        """
        )

        # Results Section
        results_layout = QVBoxLayout(results_frame)

        # Results Title
        results_title = QLabel("Prediction Results")
        results_title.setFont(QFont("Gothic", 20, QFont.Bold))
        results_title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        results_layout.addWidget(results_title)

        # Prediction Label
        self.prediction_label = QLabel("No Prediction Yet")
        self.prediction_label.setFont(QFont("Gothic", 20))
        self.prediction_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.prediction_label)

        # Confidence Label
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Gothic", 14))
        self.confidence_label.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.confidence_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        results_layout.addWidget(self.progress_bar)

        right_layout.addWidget(results_frame)

        button_layout = QHBoxLayout()

        self.clear_button = QPushButton("Clear")
        self.clear_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                font-family: "Gothic";
            }

            QPushButton:hover {
                background-color: #c0392b;
            }
        """
        )
        self.clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_button)

        right_layout.addLayout(button_layout)
        content_layout.addLayout(right_layout)
        main_layout.addLayout(content_layout)

        # Status Bar
        self.statusBar().showMessage("Load an Image to classify.")

    def load_model(self):
        try:
            model_path = MODEL_PATH

            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")

                # Check file extension to determine loading method
                if model_path.endswith(".pkl"):
                    # Load .pkl file using joblib
                    self.model = joblib.load(model_path)
                    print("Loaded .pkl model using joblib")
                elif model_path.endswith(".h5"):
                    # Load .h5 file using TensorFlow
                    self.model = tf.keras.models.load_model(model_path)
                    print("Loaded .h5 model using TensorFlow")
                else:
                    raise ValueError(f"Unsupported model format: {model_path}")

                # DEBUG: Print detailed model info
                print("=== MODEL DEBUG INFO ===")
                print(f"Model type: {type(self.model)}")

                # Try to get model info (works for both .h5 and .pkl)
                try:
                    if hasattr(self.model, "input_shape"):
                        print(f"Model input shape: {self.model.input_shape}")
                        print(f"Model output shape: {self.model.output_shape}")
                        print(
                            f"Number of classes in model: {self.model.output_shape[-1]}"
                        )
                    else:
                        print("Model info not available (likely .pkl format)")
                except Exception as e:
                    print(f"Could not get model info: {e}")

                print(f"Number of classes in our list: 15")

                # Try to get model summary (only works for TensorFlow models)
                try:
                    if hasattr(self.model, "summary"):
                        self.model.summary()
                except Exception as e:
                    print(f"Could not display model summary: {e}")
                print("========================")

                self.statusBar().showMessage("Model loaded successfully.")
            else:
                print(f"Model file not found at: {model_path}")
                self.statusBar().showMessage(f"Model file not found: {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.statusBar().showMessage(f"Error loading model: {str(e)}")

    def handle_image_drop(self, file_path):
        if not self.model:
            self.statusBar().showMessage("Model not loaded.")
            return
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(
            224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_preview.setPixmap(scaled_pixmap)
        self.image_preview.show()

        self.progress_bar.show()
        self.prediction_label.setText("Getting Name...")
        self.confidence_label.setText("Acquiring Confidence...")

        self.prediction_thread = PredictionThread(file_path, self.model)
        self.prediction_thread.prediction_ready.connect(self.display_prediction)
        self.prediction_thread.start()

    def display_prediction(self, prediction, confidence):
        self.progress_bar.hide()

        if prediction.startswith("Error"):
            self.prediction_label.setText(prediction)
            self.prediction_label.setStyleSheet("color: #e74c3c; margin: 10px;")
            self.confidence_label.setText("")
        else:
            self.prediction_label.setText(f"🐾 {prediction.title()}")
            self.prediction_label.setStyleSheet("color: #27ae60; margin: 10px;")
            self.confidence_label.setText(f"Confidence: {confidence:.2%}")

        self.statusBar().showMessage("Classification complete")

    def clear_results(self):
        self.prediction_label.setText("No prediction yet")
        self.prediction_label.setStyleSheet("color: #27ae60; margin: 10px;")
        self.confidence_label.setText("")
        self.image_preview.hide()
        self.progress_bar.hide()
        self.statusBar().showMessage("Results cleared")


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = AnimalClassificationApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
