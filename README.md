*Real-Time Crosshair Overlay*

A Python application designed for the video game **Rust** (developed by Facepunch Studios), a survival multiplayer game. This tool captures your screen, runs real-time object detection using YOLO models, and overlays detection results (enemy bounding boxes, gun classification, ranges, and a crosshair) on top of your display. Built with PyQt5 for the overlay UI and mss for screen capture.

Features:

• Gun vs. no-gun classification (binary detector)

• Gun type classification with their range values

• Enemy detection with bounding box overlay

• Semi-transparent, always-on-top overlay window

• Configurable confidence threshold and detection delay

• Crosshair drawn at the center of the screen

Project Structure

```
.
├── RealTimeCrosshairOverlay.py
└── models/
    ├── Binary_gun_nogun_detector.pt
    ├── enemy_detector.pt
    └── gun_class_detector.pt
```

• **RealTimeCrosshairOverlay.py**: Main application script
• **models/**: Pre-trained YOLO model weights

* Binary\_gun\_nogun\_detector.pt
  
* enemy\_detector.pt
  
* gun\_class\_detector.pt
  

Requirements

• Python 3.8+

• PyQt5

• mss

• ultralytics (YOLOv8)

• OpenCV (cv2)

• NumPy

Install dependencies via pip:

```bash
pip install pyqt5 mss ultralytics opencv-python numpy
```

Usage

1. Clone the repository.
   
3. Run the application:
   

   ```bash
   python RealTimeCrosshairOverlay.py
   ```
5. Launch the game **Rust**, or open the **example video** included in this repo in full-screen mode. The example video demonstrates the overlay in action, with the correct predictions shown in **pink** at the top right corner. The first 2 minutes focuses on testing the different gun types, and the section after that thoroughly tests the enemy detection model using large-group gameplay. You can also search for any gameplay video on **YouTube** to test the application, but note that there are many aspects to Rust beyond guns and enemies, so testing on random YouTube videos may not quickly cover all scenarios.

When running, the overlay window will appear on your primary monitor, showing:

• Text messages with gun predictions and ranges

• Red bounding boxes around detected enemies

• A red crosshair at the screen center

How It Works

• **Screen Capture**: Uses mss to grab the primary monitor.

• **Detection Thread**: A QThread subclass (DetectionThread) continuously:

1. Captures a frame
   
3. Runs binary (gun/no-gun) detection on a resized 360p image
   
5. If a gun is detected, classifies the gun type and retrieves range values
   
7. Runs enemy detection on the full-resolution image
   
9. Emits results via a Qt signal
   • **Overlay Window**: A QWidget subclass (OverlayWindow) listens for updates, then:

* Draws text messages (gun info)
  
* Draws enemy bounding boxes
  
* Draws a center crosshair
