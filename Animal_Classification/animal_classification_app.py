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


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Development mode: use current directory
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Model paths that work in both development and PyInstaller modes
MODEL_PATH_1 = get_resource_path("aniClass_EFF_Stage1.pkl")
MODEL_PATH_2 = get_resource_path("aniClass_EFF_Stage2.pkl")


class PredictionThread(QThread):
    prediction_ready = pyqtSignal(str, float, str)

    def __init__(self, image_path, model_stage1, model_stage2):
        super().__init__()
        self.image_path = image_path
        self.model_stage1 = model_stage1
        self.model_stage2 = model_stage2

    def run(self):
        try:
            img = Image.open(self.image_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

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

            predictions = {}
            details_info = []

            # Stage 1 Prediction
            if self.model_stage1 is not None:
                try:
                    prediction1 = self.model_stage1(img_array, training=False)
                    pred1_idx = np.argmax(prediction1, axis=1)[0]
                    conf1 = prediction1[0][pred1_idx]
                    pred1_class = animal_classes[pred1_idx]

                    predictions["stage1"] = {
                        "class": pred1_class,
                        "confidence": float(conf1),
                        "probabilities": prediction1[0],
                    }
                    details_info.append(f"Stage 1: {pred1_class} ({conf1:.2%})")

                except Exception as e:
                    details_info.append(f"Stage 1 Error: {str(e)}")

            # Stage 2 Prediction
            if self.model_stage2 is not None:
                try:
                    prediction2 = self.model_stage2(img_array, training=False)
                    pred2_idx = np.argmax(prediction2, axis=1)[0]
                    conf2 = prediction2[0][pred2_idx]
                    pred2_class = animal_classes[pred2_idx]

                    predictions["stage2"] = {
                        "class": pred2_class,
                        "confidence": float(conf2),
                        "probabilities": prediction2[0],
                    }
                    details_info.append(f"Stage 2: {pred2_class} ({conf2:.2%})")

                except Exception as e:
                    details_info.append(f"Stage 2 Error: {str(e)}")

            # Consolidate predictions
            if predictions:
                final_prediction, final_confidence, consolidation_details = (
                    self.consolidate_predictions(predictions, animal_classes)
                )
                details_info.extend(consolidation_details)
            else:
                final_prediction = "No models available"
                final_confidence = 0.0
                details_info.append("No valid predictions obtained")

            # Format details
            details_text = "\n".join(details_info)

            self.prediction_ready.emit(final_prediction, final_confidence, details_text)

        except Exception as e:
            self.prediction_ready.emit(
                f"Error: {str(e)}", 0.0, f"Processing error: {str(e)}"
            )

    def consolidate_predictions(self, predictions, animal_classes):
        """
        Consolidate predictions from multiple models using different strategies
        """
        details = []

        # Strategy 1: If both models agree on the class
        if (
            len(predictions) == 2
            and "stage1" in predictions
            and "stage2" in predictions
        ):
            stage1_pred = predictions["stage1"]
            stage2_pred = predictions["stage2"]

            if stage1_pred["class"] == stage2_pred["class"]:
                # Both models agree - use weighted average of confidence
                # Give slightly more weight to stage2 (fine-tuned model)
                consolidated_confidence = (
                    stage1_pred["confidence"] * 0.4 + stage2_pred["confidence"] * 0.6
                )
                details.append(f"Both models agree on: {stage1_pred['class']}")
                details.append(
                    f"Consolidated confidence: {consolidated_confidence:.2%}"
                )
                return stage1_pred["class"], consolidated_confidence, details
            else:
                # Models disagree - use the one with higher confidence
                if stage1_pred["confidence"] > stage2_pred["confidence"]:
                    winner = "Stage 1"
                    final_class = stage1_pred["class"]
                    final_confidence = stage1_pred["confidence"]
                else:
                    winner = "Stage 2"
                    final_class = stage2_pred["class"]
                    final_confidence = stage2_pred["confidence"]

                details.append(f"Models disagree!")
                details.append(f"Using {winner} prediction: {final_class}")
                return final_class, final_confidence, details

        # Strategy 2: Ensemble prediction (average probabilities)
        elif len(predictions) >= 2:
            # Average the probability distributions
            avg_probabilities = np.zeros(len(animal_classes))
            model_count = 0

            for stage, pred_data in predictions.items():
                avg_probabilities += pred_data["probabilities"]
                model_count += 1

            avg_probabilities /= model_count
            final_idx = np.argmax(avg_probabilities)
            final_class = animal_classes[final_idx]
            final_confidence = avg_probabilities[final_idx]

            details.append(f"Ensemble prediction from {model_count} models")
            details.append(f"Final result: {final_class} ({final_confidence:.2%})")
            return final_class, float(final_confidence), details

        # Strategy 3: Single model prediction
        elif len(predictions) == 1:
            single_pred = list(predictions.values())[0]
            stage_name = list(predictions.keys())[0]
            details.append(f"Single model prediction ({stage_name.title()})")
            return single_pred["class"], single_pred["confidence"], details

        # Fallback
        else:
            return "Unknown", 0.0, ["No valid predictions available"]


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
        self.model_stage1 = None
        self.model_stage2 = None
        self.prediction_thread = None
        self.init_ui()
        self.load_models()

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

        # Details Text Area
        self.details_text = QTextEdit()
        self.details_text.setFont(QFont("Consolas", 10))
        self.details_text.setMaximumHeight(150)
        self.details_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 5px;
            }
        """
        )
        self.details_text.hide()
        results_layout.addWidget(self.details_text)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        results_layout.addWidget(self.progress_bar)

        right_layout.addWidget(results_frame)

        # Model Status
        model_status_frame = QFrame()
        model_status_frame.setStyleSheet(
            """
            QFrame {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f8f9fa;
                padding: 10px;
                margin-top: 10px;
            }
        """
        )
        model_status_layout = QVBoxLayout(model_status_frame)

        status_title = QLabel("Model Status")
        status_title.setFont(QFont("Gothic", 14, QFont.Bold))
        model_status_layout.addWidget(status_title)

        self.model_status_label = QLabel("Loading models...")
        self.model_status_label.setFont(QFont("Gothic", 12))
        model_status_layout.addWidget(self.model_status_label)

        right_layout.addWidget(model_status_frame)

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

        self.toggle_details_button = QPushButton("Show Details")
        self.toggle_details_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                font-family: "Gothic";
            }

            QPushButton:hover {
                background-color: #2980b9;
            }
        """
        )
        self.toggle_details_button.clicked.connect(self.toggle_details)
        self.toggle_details_button.hide()  # Initially hidden
        button_layout.addWidget(self.toggle_details_button)

        right_layout.addLayout(button_layout)
        content_layout.addLayout(right_layout)
        main_layout.addLayout(content_layout)

        # Status Bar
        self.statusBar().showMessage("Loading models...")

    def load_models(self):
        """Load both Stage 1 and Stage 2 models"""
        models_loaded = []
        models_failed = []

        # Load Stage 1 Model
        try:
            if os.path.exists(MODEL_PATH_1):
                print(f"Loading Stage 1 model from: {MODEL_PATH_1}")
                if MODEL_PATH_1.endswith(".pkl"):
                    self.model_stage1 = joblib.load(MODEL_PATH_1)
                elif MODEL_PATH_1.endswith(".h5"):
                    self.model_stage1 = tf.keras.models.load_model(MODEL_PATH_1)
                models_loaded.append("Stage 1")
                print("✅ Stage 1 model loaded successfully")
            else:
                models_failed.append("Stage 1 (file not found)")
        except Exception as e:
            models_failed.append(f"Stage 1 ({str(e)})")
            print(f"❌ Failed to load Stage 1 model: {e}")

        # Load Stage 2 Model
        try:
            if os.path.exists(MODEL_PATH_2):
                print(f"Loading Stage 2 model from: {MODEL_PATH_2}")
                if MODEL_PATH_2.endswith(".pkl"):
                    self.model_stage2 = joblib.load(MODEL_PATH_2)
                elif MODEL_PATH_2.endswith(".h5"):
                    self.model_stage2 = tf.keras.models.load_model(MODEL_PATH_2)
                models_loaded.append("Stage 2")
                print("✅ Stage 2 model loaded successfully")
            else:
                models_failed.append("Stage 2 (file not found)")
        except Exception as e:
            models_failed.append(f"Stage 2 ({str(e)})")
            print(f"❌ Failed to load Stage 2 model: {e}")

        # Update status
        if models_loaded:
            status_text = f"✅ Loaded: {', '.join(models_loaded)}"
            if models_failed:
                status_text += f"\n❌ Failed: {', '.join(models_failed)}"
            self.model_status_label.setText(status_text)
            self.statusBar().showMessage(f"Models loaded: {', '.join(models_loaded)}")
        else:
            status_text = f"❌ No models loaded!\n{', '.join(models_failed)}"
            self.model_status_label.setText(status_text)
            self.statusBar().showMessage("No models available")

    def handle_image_drop(self, file_path):
        if not self.model_stage1 and not self.model_stage2:
            self.statusBar().showMessage("No models loaded.")
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

        self.prediction_thread = PredictionThread(
            file_path, self.model_stage1, self.model_stage2
        )
        self.prediction_thread.prediction_ready.connect(self.display_prediction)
        self.prediction_thread.start()

    def display_prediction(self, prediction, confidence, details):
        self.progress_bar.hide()

        if prediction.startswith("Error"):
            self.prediction_label.setText(prediction)
            self.prediction_label.setStyleSheet("color: #e74c3c; margin: 10px;")
            self.confidence_label.setText("")
        else:
            self.prediction_label.setText(f"🐾 {prediction.title()}")
            self.prediction_label.setStyleSheet("color: #27ae60; margin: 10px;")
            self.confidence_label.setText(f"Final Confidence: {confidence:.2%}")

        # Update details
        self.details_text.setText(details)
        self.toggle_details_button.show()

        self.statusBar().showMessage("Dual-stage classification complete")

    def toggle_details(self):
        if self.details_text.isVisible():
            self.details_text.hide()
            self.toggle_details_button.setText("Show Details")
        else:
            self.details_text.show()
            self.toggle_details_button.setText("Hide Details")

    def clear_results(self):
        self.prediction_label.setText("No prediction yet")
        self.prediction_label.setStyleSheet("color: #27ae60; margin: 10px;")
        self.confidence_label.setText("")
        self.image_preview.hide()
        self.progress_bar.hide()
        self.details_text.hide()
        self.details_text.clear()
        self.toggle_details_button.hide()
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
