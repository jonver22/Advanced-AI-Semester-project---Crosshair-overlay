import sys
import time
import numpy as np
import cv2
from mss import mss
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

CONFIDENCE_THRESHOLD = 0.95
DETECTION_DELAY = 0.1  

AK_range = 188
custom_smg_range = 50
hmlmg_range = 317
l96_range = 1125
lr_range = 188
mp5_range = 72
p2_range = 45
python_range = 72
thompson_range = 90
bolty_range = 574
semi_rifle_range = 188

# Overlay window using PyQt5, for detection info and crosshair overlay.
class OverlayWindow(QWidget):
    def __init__(self, geometry):
        super().__init__()
        # for overlay
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(geometry["left"], geometry["top"], geometry["width"], geometry["height"])
        self.messages = []          # List text messages.
        self.enemy_boxes = []       # List enemy bounding boxes.

    def updateMessages(self, messages, enemy_boxes):
        self.messages = messages
        self.enemy_boxes = enemy_boxes
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        # text messages.
        painter.setFont(QFont('Arial', 20))
        painter.setPen(QColor(0, 255, 0))
        y = 30
        for msg in self.messages:
            painter.drawText(10, y, msg)
            y += 30

        # enemy bounding boxes.
        painter.setPen(QColor(255, 0, 0))
        for box in self.enemy_boxes:
            x1, y1, x2, y2 = box
            width = int(x2 - x1)
            height = int(y2 - y1)
            painter.drawRect(int(x1), int(y1), width, height)

        # crosshair.
        center_x = self.width() // 2
        center_y = self.height() // 2
        cross_size = 5  # half-length cross arms (total size = 10 pixels)
        painter.setPen(QColor(255, 0, 0))
        painter.drawLine(center_x - cross_size, center_y, center_x + cross_size, center_y)
        painter.drawLine(center_x, center_y - cross_size, center_x, center_y + cross_size)

# Detection thread runs continuously and emits overlay messages and enemy boxes.
class DetectionThread(QThread):
    update_signal = pyqtSignal(list, list)  # Emits a list of messages and list of enemy boxes.

    def run(self):
        # Load models.
        binary_model = YOLO('c:/UCLL/jaar 2/S2/Advanced Ai/Semester project/code/models/Binary_gun_nogun_detector.pt')
        gun_model = YOLO('c:/UCLL/jaar 2/S2/Advanced Ai/Semester project/code/models/gun_class_detector.pt')
        enemy_model = YOLO('c:/UCLL/jaar 2/S2/Advanced Ai/Semester project/code/models/enemy_detector.pt')
        
        # Move models to GPU if available.
        binary_model.to("cuda")
        gun_model.to("cuda")
        enemy_model.to("cuda")
        
        # Initialize persistent gun message empty.
        prev_gun_msg = ""
        
        with mss() as sct:
            monitor = sct.monitors[1]  # monitor for capture.
            while True:
                messages = []
                enemy_boxes = []  # bounding boxes: each [x1, y1, x2, y2].

                # Capture screen image.
                screenshot = sct.grab(monitor)
                full_frame = np.array(screenshot)
                full_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGRA2BGR)

                # Resize full_frame to 360p for gun detection (works better this way).
                height, width = full_frame.shape[:2]
                new_height = 360
                new_width = int((new_height / height) * width)
                gun_frame = cv2.resize(full_frame, (new_width, new_height))

                # binary prediction (gun vs no gun) on the 360p gun frame.
                binary_results = binary_model.predict(source=gun_frame, conf=0.4, verbose=False)
                binary_preds = binary_results[0]
                if hasattr(binary_preds.probs, "cpu"):
                    np_probs_bin = binary_preds.probs.cpu().data.numpy()
                else:
                    np_probs_bin = np.array(binary_preds.probs.data)
                predicted_binary_idx = int(np.argmax(np_probs_bin))
                predicted_binary_class = binary_preds.names[predicted_binary_idx]
                binary_confidence = np_probs_bin[predicted_binary_idx]

                new_gun_msg = None
                # Only update message when threshold is met.
                if binary_confidence >= CONFIDENCE_THRESHOLD:
                    if predicted_binary_class.lower() == "gun":
                        gun_results = gun_model.predict(source=gun_frame, conf=0.4, verbose=False)
                        gun_preds = gun_results[0]
                        if hasattr(gun_preds.probs, "cpu"):
                            np_probs_gun = gun_preds.probs.cpu().data.numpy()
                        else:
                            np_probs_gun = np.array(gun_preds.probs.data)
                        predicted_gun_idx = int(np.argmax(np_probs_gun))
                        predicted_gun_class = gun_preds.names[predicted_gun_idx]
                        gun_confidence = np_probs_gun[predicted_gun_idx]
                        if gun_confidence >= CONFIDENCE_THRESHOLD:
                            range_value = None
                            lower_gun = predicted_gun_class.lower()
                            if lower_gun == "ak":
                                range_value = AK_range
                            elif lower_gun == "custom smg":
                                range_value = custom_smg_range
                            elif lower_gun == "hmlmg":
                                range_value = hmlmg_range
                            elif lower_gun == "l96":
                                range_value = l96_range
                            elif lower_gun == "lr":
                                range_value = lr_range
                            elif lower_gun == "mp5":
                                range_value = mp5_range
                            elif lower_gun == "p2":
                                range_value = p2_range
                            elif lower_gun == "python":
                                range_value = python_range
                            elif lower_gun == "thompson":
                                range_value = thompson_range
                            elif lower_gun == "bolty":
                                range_value = bolty_range
                            elif lower_gun == "semi rifle":
                                range_value = semi_rifle_range

                            if range_value is not None:
                                new_gun_msg = f"Predicted gun: {predicted_gun_class} ({gun_confidence:.2f}) | Range: {range_value}"
                            else:
                                new_gun_msg = f"Predicted gun: {predicted_gun_class} ({gun_confidence:.2f})"
                    else:
                        new_gun_msg = f"No gun (binary confidence: {binary_confidence:.2f})"

                # If no new prediction meets threshold, keep previous message.
                if new_gun_msg is None:
                    new_gun_msg = prev_gun_msg
                else:
                    prev_gun_msg = new_gun_msg

                messages.append(new_gun_msg)

                # enemy detection (using full quality image, needed for enemies far away).
                enemy_results = enemy_model.predict(source=full_frame, conf=0.4, verbose=False)
                enemy_preds = enemy_results[0]
                if enemy_preds.boxes is not None and len(enemy_preds.boxes) > 0:
                    boxes = enemy_preds.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        enemy_boxes.append(box.tolist())

                # Emit updated messages and enemy boxes.
                self.update_signal.emit(messages, enemy_boxes)
                time.sleep(DETECTION_DELAY)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    with mss() as sct:
        monitor = sct.monitors[1]
        monitor_geometry = {
            "left": monitor["left"],
            "top": monitor["top"],
            "width": monitor.get("width", 800),
            "height": monitor.get("height", 600)
        }
    overlay = OverlayWindow(monitor_geometry)
    overlay.show()

    detection_thread = DetectionThread()
    detection_thread.update_signal.connect(overlay.updateMessages)
    detection_thread.start()

    sys.exit(app.exec_())