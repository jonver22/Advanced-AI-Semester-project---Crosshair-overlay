Real-Time Crosshair Overlay

A Python application that captures your screen, runs real-time object detection using YOLO models, and overlays detection results (enemy bounding boxes, gun classification, ranges, and a crosshair) on top of your display. Built with PyQt5 for the overlay UI and mss for screen capture.

Features

•	Gun vs. no-gun classification (binary detector)

•	Gun type classification with their range values

•	Enemy detection with bounding box overlay

•	Semi-transparent, always-on-top overlay window

•	Configurable confidence threshold and detection delay

•	Crosshair drawn at the center of the screen


Project Structure

.
├── RealTimeCrosshairOverlay.py

└── models/

 ├── Binary_gun_nogun_detector.pt
    
 ├── custom_classification_model_binary_gun_nogun.pt
    
 ├── custom_classification_model_gun_detector.pt
    
 ├── enemy_detector.pt
 
 └── gun_class_detector.pt


•	RealTimeCrosshairOverlay.py: Main application script

•	models/: Pre-trained YOLO model weights

   -	Binary_gun_nogun_detector.pt

   -    custom_classification_model_binary_gun_nogun.pt

   -	custom_classification_model_gun_detector.pt

   -	enemy_detector.pt

   -	gun_class_detector.pt


Requirements

•	Python 3.8+

•	PyQt5

•	mss

•	ultralytics (YOLOv8)

•	OpenCV (cv2)

•	NumPy


Install dependencies via pip:


pip install pyqt5 mss ultralytics opencv-python numpy


Usage

1.	Clone the repository and place all model files in the models directory.
   
2.	Run the application:
   
python RealTimeCrosshairOverlay.py


The overlay window will appear on your primary monitor, showing:

•	Text messages with gun predictions and ranges

•	Red bounding boxes around detected enemies

•	A red crosshair at the screen center


How It Works

•	Screen Capture: Uses mss to grab the primary monitor.

•	Detection Thread: A QThread subclass (DetectionThread) continuously:

  1.	Captures a frame
   
  2.	Runs binary (gun/no-gun) detection on a resized 360p image
   
  3.	If gun is detected, classifies the gun type and computes range
   
  4.	Runs enemy detection on the full-resolution image
   
  5.	Emits results via a Qt signal
    
•	Overlay Window: A QWidget subclass (OverlayWindow) listens for updates, then:

     o	Draws text messages (gun info)

     o	Draws enemy bounding boxes

     o	Draws a center crosshair
