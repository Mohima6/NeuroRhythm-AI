import sys
import os

# ==============================================
# CRITICAL: Set these BEFORE any imports
# ==============================================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# Import all required packages
# ==============================================
import cv2
import numpy as np
from collections import deque
import time
from scipy import signal, stats
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from datetime import datetime

# Import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe imported successfully")
except Exception as e:
    print(f"✗ MediaPipe import error: {e}")
    MEDIAPIPE_AVAILABLE = False
    # Exit if MediaPipe is not available
    sys.exit(1)


class NeuroVision(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            print("MediaPipe not available")
            sys.exit(1)
        
        # Initialize tracking variables
        self.initialize_tracking_arrays()
        
        # Setup GUI
        self.setup_ui()
        
        # Start camera
        self.start_camera()

    def initialize_tracking_arrays(self):
        """Initialize all tracking arrays and variables"""
        # Eye tracking
        self.eye_aspect_ratios = deque(maxlen=30)
        self.blink_count = 0
        self.blink_times = deque(maxlen=100)
        self.last_blink_time = time.time()

        # Gaze tracking
        self.gaze_history = deque(maxlen=30)
        self.gaze_stability = 0

        # Facial expression tracking
        self.brow_tension_history = deque(maxlen=30)
        self.smile_intensity_history = deque(maxlen=30)
        self.face_tension_history = deque(maxlen=30)

        # State tracking
        self.focus_scores = deque(maxlen=100)
        self.stress_scores = deque(maxlen=100)
        self.drowsiness_scores = deque(maxlen=100)
        self.distraction_events = []

        # Metrics
        self.session_start = time.time()
        self.productivity_score = 0

        # Task classification
        self.current_task = "Unknown"

    def setup_ui(self):
        """Setup the PyQt5 GUI"""
        self.setWindowTitle("NeuroRhythm AI - Vision Module")
        self.setGeometry(100, 100, 1400, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Camera feed and metrics
        left_panel = QVBoxLayout()

        # Camera feed
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid #2c3e50; background-color: #1a1a1a;")
        left_panel.addWidget(self.camera_label)

        # Real-time metrics
        metrics_group = QGroupBox("Real-Time Metrics")
        metrics_layout = QGridLayout()

        # Focus metrics
        self.focus_label = QLabel("Focus: 0%")
        self.focus_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2ecc71;")
        metrics_layout.addWidget(QLabel("🎯 Focus:"), 0, 0)
        metrics_layout.addWidget(self.focus_label, 0, 1)

        # Stress metrics
        self.stress_label = QLabel("Stress: 0%")
        self.stress_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c;")
        metrics_layout.addWidget(QLabel("😰 Stress:"), 1, 0)
        metrics_layout.addWidget(self.stress_label, 1, 1)

        # Alertness metrics
        self.alertness_label = QLabel("Alertness: 100%")
        self.alertness_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #3498db;")
        metrics_layout.addWidget(QLabel("😴 Alertness:"), 2, 0)
        metrics_layout.addWidget(self.alertness_label, 2, 1)

        # Blink rate
        self.blink_label = QLabel("Blink Rate: 0/min")
        self.blink_label.setStyleSheet("font-size: 14px; color: #f39c12;")
        metrics_layout.addWidget(QLabel("👁 Blink Rate:"), 3, 0)
        metrics_layout.addWidget(self.blink_label, 3, 1)

        metrics_group.setLayout(metrics_layout)
        left_panel.addWidget(metrics_group)

        # Right panel - Charts
        right_panel = QVBoxLayout()

        # Charts
        charts_group = QGroupBox("Real-Time Monitoring")
        charts_layout = QVBoxLayout()

        # Focus trend chart
        self.focus_plot = pg.PlotWidget(title="Focus Trend")
        self.focus_plot.setLabel('left', 'Focus (%)')
        self.focus_plot.setLabel('bottom', 'Time')
        self.focus_plot.setYRange(0, 100)
        self.focus_plot.setBackground('#1a1a1a')
        self.focus_curve = self.focus_plot.plot(pen=pg.mkPen(color='#2ecc71', width=2))
        charts_layout.addWidget(self.focus_plot)

        # Stress trend chart
        self.stress_plot = pg.PlotWidget(title="Stress Trend")
        self.stress_plot.setLabel('left', 'Stress (%)')
        self.stress_plot.setLabel('bottom', 'Time')
        self.stress_plot.setYRange(0, 100)
        self.stress_plot.setBackground('#1a1a1a')
        self.stress_curve = self.stress_plot.plot(pen=pg.mkPen(color='#e74c3c', width=2))
        charts_layout.addWidget(self.stress_plot)

        charts_group.setLayout(charts_layout)
        charts_group.setFixedHeight(500)
        right_panel.addWidget(charts_group)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 40)
        main_layout.addLayout(right_panel, 60)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System Ready - Tracking Active")

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # Update every 100ms (10 FPS)

    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera started successfully")

    def calculate_eye_aspect_ratio(self, landmarks, eye_points):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        try:
            # Get the eye landmark coordinates
            eye_coords = np.array([(landmarks[i].x, landmarks[i].y) 
                                  for i in eye_points])

            # Calculate vertical distances
            vertical1 = np.linalg.norm(eye_coords[1] - eye_coords[5])
            vertical2 = np.linalg.norm(eye_coords[2] - eye_coords[4])

            # Calculate horizontal distance
            horizontal = np.linalg.norm(eye_coords[0] - eye_coords[3])

            # Calculate EAR
            ear = (vertical1 + vertical2) / (2.0 * horizontal)
            return ear
        except:
            return 0.3  # Default value

    def detect_blinks(self, left_ear, right_ear):
        """Detect blinks using EAR threshold"""
        ear_threshold = 0.2
        ear = (left_ear + right_ear) / 2.0

        self.eye_aspect_ratios.append(ear)

        if len(self.eye_aspect_ratios) > 1:
            if ear < ear_threshold and self.eye_aspect_ratios[-2] >= ear_threshold:
                self.blink_count += 1
                self.blink_times.append(time.time())
                self.last_blink_time = time.time()
                return True
        return False

    def calculate_blink_rate(self):
        """Calculate blink rate per minute"""
        if len(self.blink_times) < 2:
            return 0

        # Use last 30 seconds
        recent_times = [t for t in self.blink_times if time.time() - t < 30]
        if len(recent_times) < 2:
            return 0

        intervals = np.diff(sorted(recent_times))
        if len(intervals) > 0:
            return 60 / np.mean(intervals)
        return 0

    def process_frame(self, frame):
        """Process a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        focus_score = 0
        stress_score = 0
        alertness = 100
        blink_rate = 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Eye aspect ratio for blink detection
            left_eye_points = [33, 160, 158, 133, 153, 144]
            right_eye_points = [362, 385, 387, 263, 373, 380]
            
            left_ear = self.calculate_eye_aspect_ratio(landmarks, left_eye_points)
            right_ear = self.calculate_eye_aspect_ratio(landmarks, right_eye_points)
            
            blink_detected = self.detect_blinks(left_ear, right_ear)
            blink_rate = self.calculate_blink_rate()
            
            # Calculate metrics (simplified for demo)
            ear = (left_ear + right_ear) / 2
            focus_score = min(100, max(0, (ear - 0.2) * 500))  # Simple focus calculation
            stress_score = min(100, max(0, (0.4 - ear) * 333))  # Simple stress calculation
            alertness = max(0, min(100, 100 - ((0.25 - ear) * 400)))  # Alertness
            
            # Update history
            self.focus_scores.append(focus_score)
            self.stress_scores.append(stress_score)
            self.drowsiness_scores.append(alertness)
            
            # Draw landmarks
            h, w = frame.shape[:2]
            for landmark in landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Draw eye landmarks
            for point in left_eye_points + right_eye_points:
                x, y = int(landmarks[point].x * w), int(landmarks[point].y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            
            # Add metrics overlay
            cv2.putText(frame, f"Focus: {focus_score:.1f}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Stress: {stress_score:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Alertness: {alertness:.1f}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if blink_detected:
                cv2.putText(frame, "BLINK!", (w - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        else:
            # No face detected
            cv2.putText(frame, "NO FACE DETECTED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame, focus_score, stress_score, alertness, blink_rate

    def update_ui(self):
        """Update the GUI"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Process frame
        processed_frame, focus, stress, alertness, blink_rate = self.process_frame(frame)

        # Convert for display
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Update camera feed
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

        # Update metrics
        self.focus_label.setText(f"Focus: {focus:.1f}%")
        self.stress_label.setText(f"Stress: {stress:.1f}%")
        self.alertness_label.setText(f"Alertness: {alertness:.1f}%")
        self.blink_label.setText(f"Blink Rate: {blink_rate:.1f}/min")

        # Update charts
        self.update_charts()

        # Update status bar
        elapsed = time.time() - self.session_start
        self.status_bar.showMessage(
            f"NeuroRhythm AI | Session: {int(elapsed//60)}m {int(elapsed%60)}s | "
            f"Blinks: {self.blink_count}"
        )

    def update_charts(self):
        """Update all charts"""
        # Focus chart
        if len(self.focus_scores) > 0:
            x = list(range(len(self.focus_scores)))
            self.focus_curve.setData(x, list(self.focus_scores))

        # Stress chart
        if len(self.stress_scores) > 0:
            x = list(range(len(self.stress_scores)))
            self.stress_curve.setData(x, list(self.stress_scores))

    def closeEvent(self, event):
        """Clean up on close"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark theme
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 40))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(30, 30, 40))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(40, 40, 50))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

    window = NeuroVision()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
