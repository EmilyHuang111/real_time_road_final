# Project Name: Robot Control and Lane Detection System

### Introduction
This project is a GUI-based Python application designed to control a robotic car remotely, provide video feed processing, and log activities in real-time. It includes functionalities for user registration, login, and movement control (e.g., forward, backward, left, right) of the robot. It also includes a lane detection module using OpenCV for overlaying lane lines on a video feed, useful for applications in self-driving technologies and robot navigation.

### Features
- **User Registration and Login**: Allows users to create and manage accounts.
- **Robotic Control**: Control robot movements via a Flask API for forward, backward, left, and right movements.
- **Lane Detection**: Uses OpenCV to detect and highlight lane lines and curvature.
- **Logging**: Activity logging with timestamps, including login, logout, and movement actions.
- **Real-Time Video Feed**: Displays video feed from a webcam, with lane line overlays and curvature information.

### Requirements
- Python 3.6 or higher
- Modules:
  - tkinter
  - sqlite3
  - OpenCV (`cv2`)
  - PIL (`Pillow`)
  - requests
  - numpy
  - pytz

Install requirements with:
```bash
pip install opencv-python-headless pillow requests numpy pytz
