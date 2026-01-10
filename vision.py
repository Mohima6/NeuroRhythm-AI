#blink tracker
#face detector
#gaze estimator
#stress estimator
#task classifier
#emotion tracker
#activity classifier
#keyboard monitor
#mouse monitor
#brain audio environment




import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from scipy import signal, stats
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class NeuroVision(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize tracking variables
        self.initialize_tracking_arrays()

        # Setup GUI
        self.setup_ui()

        # Start camera thread
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
        self.session_data = {
            'focus_periods': [],
            'distraction_periods': [],
            'stress_peaks': [],
            'blink_patterns': []
        }

        # Task classification
        self.current_task = "Unknown"
        self.task_patterns = []

    def setup_ui(self):
        """Setup the PyQt5 GUI"""
        self.setWindowTitle("NeuroRhythm AI - Vision Module")
        self.setGeometry(100, 100, 1600, 900)

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
        self.camera_label.setStyleSheet("border: 2px solid #2c3e50;")
        left_panel.addWidget(self.camera_label)

        # Real-time metrics
        metrics_group = QGroupBox("Real-Time Metrics")
        metrics_layout = QGridLayout()

        # Focus metrics
        self.focus_label = QLabel("Focus: 0%")
        self.focus_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        metrics_layout.addWidget(QLabel("🎯 Focus:"), 0, 0)
        metrics_layout.addWidget(self.focus_label, 0, 1)

        # Stress metrics
        self.stress_label = QLabel("Stress: 0%")
        self.stress_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        metrics_layout.addWidget(QLabel("😰 Stress:"), 1, 0)
        metrics_layout.addWidget(self.stress_label, 1, 1)

        # Drowsiness metrics
        self.drowsiness_label = QLabel("Alertness: 100%")
        self.drowsiness_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        metrics_layout.addWidget(QLabel("😴 Drowsiness:"), 2, 0)
        metrics_layout.addWidget(self.drowsiness_label, 2, 1)

        # Blink rate
        self.blink_label = QLabel("Blink Rate: 0/min")
        metrics_layout.addWidget(QLabel("👁 Blink Rate:"), 3, 0)
        metrics_layout.addWidget(self.blink_label, 3, 1)

        # Task classification
        self.task_label = QLabel("Task: Unknown")
        self.task_label.setStyleSheet("font-size: 14px; color: #3498db;")
        metrics_layout.addWidget(QLabel("💼 Task:"), 4, 0)
        metrics_layout.addWidget(self.task_label, 4, 1)

        # Cognitive state
        self.state_label = QLabel("State: Neutral")
        self.state_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(QLabel("🧠 Cognitive State:"), 5, 0)
        metrics_layout.addWidget(self.state_label, 5, 1)

        metrics_group.setLayout(metrics_layout)
        left_panel.addWidget(metrics_group)

        # Right panel - Charts and detailed metrics
        right_panel = QVBoxLayout()

        # Charts
        charts_group = QGroupBox("Real-Time Monitoring")
        charts_layout = QVBoxLayout()

        # Focus trend chart
        self.focus_plot = pg.PlotWidget(title="Focus Trend")
        self.focus_plot.setLabel('left', 'Focus (%)')
        self.focus_plot.setLabel('bottom', 'Time')
        self.focus_plot.setYRange(0, 100)
        self.focus_curve = self.focus_plot.plot(pen=pg.mkPen(color='g', width=2))
        charts_layout.addWidget(self.focus_plot)

        # Stress trend chart
        self.stress_plot = pg.PlotWidget(title="Stress Trend")
        self.stress_plot.setLabel('left', 'Stress (%)')
        self.stress_plot.setLabel('bottom', 'Time')
        self.stress_plot.setYRange(0, 100)
        self.stress_curve = self.stress_plot.plot(pen=pg.mkPen(color='r', width=2))
        charts_layout.addWidget(self.stress_plot)

        # Drowsiness trend chart
        self.drowsiness_plot = pg.PlotWidget(title="Alertness Trend")
        self.drowsiness_plot.setLabel('left', 'Alertness (%)')
        self.drowsiness_plot.setLabel('bottom', 'Time')
        self.drowsiness_plot.setYRange(0, 100)
        self.drowsiness_curve = self.drowsiness_plot.plot(pen=pg.mkPen(color='b', width=2))
        charts_layout.addWidget(self.drowsiness_plot)

        charts_group.setLayout(charts_layout)
        charts_group.setFixedHeight(600)
        right_panel.addWidget(charts_group)

        # Feature indicators
        features_group = QGroupBox("Active Features")
        features_layout = QGridLayout()

        # Feature status indicators
        features = [
            ("🧠 Neural Phase-Locking", True, "#2ecc71"),
            ("🎵 AI Neuro-Music", True, "#3498db"),
            ("🔄 RL Adaptive Audio", True, "#e74c3c"),
            ("👁 Non-Contact Brain State", True, "#9b59b6"),
            ("📊 Multi-Modal Profiling", True, "#1abc9c"),
            ("⚡ Neural Reset", False, "#f39c12"),
            ("🌿 Hybrid Entrainment", True, "#27ae60"),
            ("📈 Cognitive Analytics", True, "#8e44ad"),
            ("💻 Developer API", False, "#d35400"),
            ("🔮 Predictive Intelligence", True, "#16a085")
        ]

        for i, (feature, active, color) in enumerate(features):
            label = QLabel(feature)
            if active:
                label.setStyleSheet(f"color: {color}; font-weight: bold;")
            else:
                label.setStyleSheet("color: #95a5a6;")
            features_layout.addWidget(label, i // 2, i % 2)

        features_group.setLayout(features_layout)
        right_panel.addWidget(features_group)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 40)
        main_layout.addLayout(right_panel, 60)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("NeuroRhythm AI Vision System - Ready")

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(50)  # 20 FPS

    def start_camera(self):
        """Start camera capture in separate thread"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def calculate_eye_aspect_ratio(self, landmarks, eye_points):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
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

        recent_times = [t for t in self.blink_times
                        if time.time() - t < 60]  # Last minute
        if len(recent_times) < 2:
            return 0

        intervals = np.diff(sorted(recent_times))
        if len(intervals) > 0:
            return 60 / np.mean(intervals)
        return 0

    def analyze_facial_expressions(self, landmarks):
        """Analyze facial expressions for stress detection"""
        # Brow furrow (stress indicator)
        left_brow = np.mean([(landmarks[i].y) for i in [70, 63, 105]])
        right_brow = np.mean([(landmarks[i].y) for i in [300, 293, 334]])
        brow_tension = abs(left_brow - right_brow) * 1000

        # Smile intensity (relaxation indicator)
        mouth_width = abs(landmarks[61].x - landmarks[291].x)
        mouth_height = abs(landmarks[13].y - landmarks[14].y)
        smile_intensity = mouth_width / mouth_height if mouth_height > 0 else 0

        # Facial tension (from jaw clenching)
        jaw_tension = abs(landmarks[152].y - landmarks[175].y)

        self.brow_tension_history.append(brow_tension)
        self.smile_intensity_history.append(smile_intensity)
        self.face_tension_history.append(jaw_tension)

        return brow_tension, smile_intensity, jaw_tension

    def estimate_gaze(self, landmarks):
        """Estimate gaze direction"""
        # Simplified gaze estimation using eye landmarks
        left_eye_center = np.mean([(landmarks[i].x, landmarks[i].y)
                                   for i in [33, 133]], axis=0)
        right_eye_center = np.mean([(landmarks[i].x, landmarks[i].y)
                                    for i in [362, 263]], axis=0)

        # Calculate gaze deviation
        face_center_x = 0.5  # Assuming normalized coordinates
        gaze_deviation = ((left_eye_center[0] + right_eye_center[0]) / 2) - face_center_x

        self.gaze_history.append(abs(gaze_deviation))

        # Calculate gaze stability (lower values = more stable)
        if len(self.gaze_history) > 5:
            self.gaze_stability = np.std(self.gaze_history)

        return gaze_deviation

    def detect_drowsiness(self, ear):
        """Detect drowsiness using PERCLOS metric"""
        # PERCLOS: Percentage of eyelid closure over time
        ear_threshold = 0.25
        if len(self.eye_aspect_ratios) < 30:
            return 0

        # Calculate percentage of time eyes were closed
        closed_frames = sum(1 for ratio in list(self.eye_aspect_ratios)[-30:]
                            if ratio < ear_threshold)
        perclos = (closed_frames / 30) * 100

        # Additional drowsiness indicators
        blink_rate = self.calculate_blink_rate()
        if blink_rate < 5 or blink_rate > 30:  # Abnormal blink rates
            perclos += 20

        return min(perclos, 100)

    def classify_task(self, gaze_stability, blink_rate, facial_tension):
        """Classify current task based on behavior patterns"""
        # Simple rule-based task classification
        if gaze_stability < 0.02 and blink_rate < 15:
            return "Deep Focus (Coding/Reading)"
        elif gaze_stability < 0.05 and 15 <= blink_rate < 25:
            return "Creative Work"
        elif gaze_stability > 0.1 and blink_rate > 25:
            return "Distracted/Browsing"
        elif facial_tension > 0.5:
            return "Stressed/Typing"
        else:
            return "Neutral/Idle"

    def calculate_focus_score(self, gaze_stability, blink_rate, ear_variability):
        """Calculate focus score (0-100%)"""
        # Normalize inputs
        gaze_score = max(0, 100 - (gaze_stability * 1000))
        blink_score = max(0, 100 - abs(blink_rate - 15) * 2)  # Optimal ~15 blinks/min
        ear_score = 100 - (ear_variability * 100)

        # Weighted average
        focus_score = (gaze_score * 0.4 + blink_score * 0.3 + ear_score * 0.3)
        return max(0, min(100, focus_score))

    def calculate_stress_score(self, brow_tension, facial_tension, blink_rate):
        """Calculate stress score (0-100%)"""
        # Normalize inputs
        brow_score = min(100, brow_tension * 10)
        face_score = min(100, facial_tension * 20)
        blink_score = min(100, max(0, blink_rate - 10) * 2)

        # Weighted average
        stress_score = (brow_score * 0.4 + face_score * 0.4 + blink_score * 0.2)
        return max(0, min(100, stress_score))

    def process_frame(self, frame):
        """Process a single frame and extract all features"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Feature 1: Blink detection
            left_eye_points = [33, 160, 158, 133, 153, 144]
            right_eye_points = [362, 385, 387, 263, 373, 380]
            left_ear = self.calculate_eye_aspect_ratio(landmarks, left_eye_points)
            right_ear = self.calculate_eye_aspect_ratio(landmarks, right_eye_points)
            blink_detected = self.detect_blinks(left_ear, right_ear)

            # Feature 2: Blink rate calculation
            blink_rate = self.calculate_blink_rate()

            # Feature 3: Facial expression analysis
            brow_tension, smile_intensity, facial_tension = \
                self.analyze_facial_expressions(landmarks)

            # Feature 4: Gaze estimation
            gaze_deviation = self.estimate_gaze(landmarks)

            # Feature 5: Drowsiness detection
            ear = (left_ear + right_ear) / 2
            drowsiness = self.detect_drowsiness(ear)

            # Feature 6: Focus score calculation
            gaze_stability = self.gaze_stability
            ear_variability = np.std(list(self.eye_aspect_ratios)[-10:]) if len(self.eye_aspect_ratios) >= 10 else 0.1
            focus_score = self.calculate_focus_score(gaze_stability, blink_rate, ear_variability)

            # Feature 7: Stress score calculation
            stress_score = self.calculate_stress_score(brow_tension, facial_tension, blink_rate)

            # Feature 8: Task classification
            self.current_task = self.classify_task(gaze_stability, blink_rate, facial_tension)

            # Feature 9: Update metrics
            self.focus_scores.append(focus_score)
            self.stress_scores.append(stress_score)
            self.drowsiness_scores.append(100 - drowsiness)  # Convert to alertness

            # Feature 10: Distraction detection
            if gaze_stability > 0.1 and focus_score < 30:
                if not hasattr(self, 'last_distraction_time') or \
                        time.time() - self.last_distraction_time > 10:
                    self.distraction_events.append(time.time())
                    self.last_distraction_time = time.time()

            # Draw landmarks and annotations on frame
            frame = self.draw_annotations(frame, landmarks, focus_score,
                                          stress_score, drowsiness, blink_detected)

            return frame, focus_score, stress_score, drowsiness, blink_rate

        return frame, 0, 0, 0, 0

    def draw_annotations(self, frame, landmarks, focus, stress, drowsiness, blink_detected):
        """Draw annotations and landmarks on the frame"""
        h, w = frame.shape[:2]

        # Draw facial landmarks
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw eye regions
        eye_points = [33, 133, 362, 263]  # Key eye landmarks
        for point in eye_points:
            x, y = int(landmarks[point].x * w), int(landmarks[point].y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        # Add status text
        cv2.putText(frame, f"Focus: {focus:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Stress: {stress:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Alertness: {100 - drowsiness:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Blink indicator
        if blink_detected:
            cv2.putText(frame, "BLINK!", (w - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Task indicator
        cv2.putText(frame, f"Task: {self.current_task}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw focus indicator bar
        focus_bar_width = int(focus * 2)
        cv2.rectangle(frame, (10, h - 50), (210, h - 30), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 50), (10 + focus_bar_width, h - 30),
                      (0, 255, 0), -1)

        # Draw stress indicator bar
        stress_bar_width = int(stress * 2)
        cv2.rectangle(frame, (10, h - 80), (210, h - 60), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 80), (10 + stress_bar_width, h - 60),
                      (0, 0, 255), -1)

        return frame

    def get_cognitive_state(self):
        """Determine overall cognitive state"""
        if len(self.focus_scores) == 0:
            return "Neutral", "#95a5a6"

        focus = np.mean(list(self.focus_scores)[-10:]) if len(self.focus_scores) >= 10 else 50
        stress = np.mean(list(self.stress_scores)[-10:]) if len(self.stress_scores) >= 10 else 30

        if focus > 70 and stress < 30:
            return "In Flow", "#2ecc71"  # Green
        elif focus > 50 and stress < 50:
            return "Focused", "#3498db"  # Blue
        elif stress > 60:
            return "Stressed", "#e74c3c"  # Red
        elif focus < 30:
            return "Distracted", "#f39c12"  # Orange
        else:
            return "Neutral", "#95a5a6"  # Gray

    def update_ui(self):
        """Update the GUI with latest data"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Process frame
        processed_frame, focus, stress, drowsiness, blink_rate = self.process_frame(frame)

        # Convert frame for display
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Update camera feed
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

        # Update metrics
        self.focus_label.setText(f"Focus: {focus:.1f}%")
        self.stress_label.setText(f"Stress: {stress:.1f}%")
        self.drowsiness_label.setText(f"Alertness: {100 - drowsiness:.1f}%")
        self.blink_label.setText(f"Blink Rate: {blink_rate:.1f}/min")
        self.task_label.setText(f"Task: {self.current_task}")

        # Update cognitive state
        state, color = self.get_cognitive_state()
        self.state_label.setText(f"State: {state}")
        self.state_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Update charts
        self.update_charts()

        # Update status bar
        elapsed = time.time() - self.session_start
        self.status_bar.showMessage(
            f"NeuroRhythm AI | Session: {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"Blinks: {self.blink_count} | Distractions: {len(self.distraction_events)}"
        )

    def update_charts(self):
        """Update all charts with new data"""
        # Focus chart
        if len(self.focus_scores) > 0:
            x = list(range(len(self.focus_scores)))
            self.focus_curve.setData(x, list(self.focus_scores))

        # Stress chart
        if len(self.stress_scores) > 0:
            x = list(range(len(self.stress_scores)))
            self.stress_curve.setData(x, list(self.stress_scores))

        # Drowsiness chart (alertness)
        if len(self.drowsiness_scores) > 0:
            x = list(range(len(self.drowsiness_scores)))
            self.drowsiness_curve.setData(x, list(self.drowsiness_scores))

    def closeEvent(self, event):
        """Clean up when closing the window"""
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

