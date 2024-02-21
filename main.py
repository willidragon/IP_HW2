# main.py

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap

from normal_prediction import run_ground_truth, run_prediction, process_and_evaluate_image
from segmentation_predict import segmentation

class ImageAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MainWindow')

        # Main layout
        layout = QVBoxLayout()

        # Load folder button
        self.load_folder_button = QPushButton('Load Folder')
        layout.addWidget(self.load_folder_button)

        # Previous and Next buttons
        self.prev_button = QPushButton('Previous')
        self.next_button = QPushButton('Next')
        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)

        # Current Image label
        self.current_image_path = None
        self.current_image_label = QLabel('Current Image: File name')
        self.current_json_path = None
        self.current_json_label = QLabel('Current JSON: File name')

        layout.addWidget(self.current_image_label)
        layout.addWidget(self.current_json_label)

        # Detection group
        self.detection_button = QPushButton('Detection')
        self.detection_button.clicked.connect(self.perform_detection)
        


        self.iou_label = QLabel('IoU :')
        self.accuracy_label = QLabel('Accuracy :')
        self.precision_label = QLabel('Precision :')
        self.recall_label = QLabel('Recall :')
        layout.addWidget(self.detection_button)
        layout.addWidget(self.iou_label)
        layout.addWidget(self.accuracy_label)
        layout.addWidget(self.precision_label)
        layout.addWidget(self.recall_label)

        # Segmentation group
        self.segmentation_button = QPushButton('Segmentation')
        self.segmentation_button.clicked.connect(self.segmentation)
        self.predicted_label = QLabel('Predicted Label :')
        self.dice_coefficient_label = QLabel('Dice Coefficient :')
        layout.addWidget(self.segmentation_button)
        layout.addWidget(self.dice_coefficient_label)
        layout.addWidget(self.predicted_label)

        # Image display
        self.image_display_label = QLabel()
        layout.addWidget(self.image_display_label)

        # Set main layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Additional variables for image handling
        self.current_image_index = 0
        self.image_folder = None
        self.image_paths = []

        # Connect button actions
        self.load_folder_button.clicked.connect(self.load_folder)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.image_folder = folder_path
            self.image_paths = [os.path.join(self.image_folder, filename) for filename in os.listdir(self.image_folder) if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            self.current_image_index = 0
            self.show_current_image()

    def show_previous_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.show_current_image()

    def show_next_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.show_current_image()

    def show_current_image(self):
        if self.image_paths:
            image_path = self.image_paths[self.current_image_index]
            pixmap = QPixmap(image_path)
            self.image_display_label.setPixmap(pixmap)
            self.current_image_label.setText(f'Current Image: {image_path}')
            self.current_json_label.setText(f'Current JSON: {image_path[:-4] + ".json"}')
            self.current_image_path = image_path
            self.current_json_path = image_path[:-4] + ".json"

    def perform_detection(self):
        if self.current_image_path and self.current_json_path:
            # Display ground truth image with bounding boxes
            run_ground_truth(self.current_json_path, self.current_image_path)

            # Display predicted image with bounding boxes
            run_prediction(self.current_image_path)

            # Evaluate metrics and update the GUI
            precision, recall, accuracy, iou = process_and_evaluate_image(
                self.current_json_path, self.current_image_path,
                original_size=(512, 512), transformed_size=(256, 256)
            )

            self.iou_label.setText(f'IoU : {round(iou, 5)}')
            self.accuracy_label.setText(f'Accuracy : {round(accuracy, 5)}')
            self.precision_label.setText(f'Precision : {round(precision, 5)}')
            self.recall_label.setText(f'Recall : {round(recall, 5)}')

    def segmentation(self):

        predicted_label, dice_coefficient = segmentation(self.current_image_path)

        self.dice_coefficient_label.setText(f'Dice Coefficient : {round(dice_coefficient, 5)}')
        self.predicted_label.setText(f'Predicted Label : {predicted_label}')
        
        pass
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = ImageAnalysisUI()
    mainWin.show()
    sys.exit(app.exec_())
