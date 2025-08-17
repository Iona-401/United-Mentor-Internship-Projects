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
from PIL import Image
import io


class PredictionThread(QThread):
    prediction_ready = pyqtSignal(str, float)

    def __init__(self, image_path, model):
        super().__init__()
        self.image_path - image_path
        self.model = model

    def run(self):
        try:
            img = Image.open(self.image_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = self.model.predict(img_array)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class_idx]

            animal_classes = [
                "cat",
                "dog",
                "bird",
                "fish",
                "horse",
                "cow",
                "sheep",
                "pig",
                "chicken",
                "duck",
                "rabbit",
                "elephant",
                "lion",
                "tiger",
                "bear",
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
                background-color: #white;
                padding: 20px;
            }
        """
        )

        results_layout = QVBoxLayout(results_frame)

        results_title = QLabel("Prediction Results")
        results_title.setFont(QFont("Gothic", 20, QFont.Bold))
        results_title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        results_layout.addWidget(results_title)

        self.prediction_label = QLabel("No Prediction Yet")
        self.prediction_label.setFont(QFont("Gothic", 20))
        self.prediction_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Gothic", 14))
        self.confidence_label.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.confidence_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
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
