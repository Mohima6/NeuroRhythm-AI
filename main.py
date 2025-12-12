"""
NEURO-RHYTHM AI - MAIN DASHBOARD
File: main.py
Description: PyQt5 GUI dashboard connecting Vision, RL, and Audio modules
"""

import sys
import os
import time
import threading
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg

# Add current directory to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from vision_engine.vision import NeuroVision

    VISION_AVAILABLE = True
except ImportError:
    print("⚠️ Vision module not found. Running without camera.")
    VISION_AVAILABLE = False

try:
    from vision_engine.rl import NeuroRLAgent, BRAINWAVE_PRESETS

    RL_AVAILABLE = True
except ImportError:
    print("⚠️ RL module not found. Running without reinforcement learning.")
    RL_AVAILABLE = False

try:
    from vision_engine.audio import AudioTherapist, AUDIO_PRESETS, CognitiveAudioMapper

    AUDIO_AVAILABLE = True
except ImportError:
    print("⚠️ Audio module not found. Running without audio.")
    AUDIO_AVAILABLE = False

# Set dark theme colors
DARK_BG = "#1a1a2e"
DARK_PANEL = "#16213e"
DARK_TEXT = "#ffffff"
ACCENT_BLUE = "#0ea5e9"
ACCENT_GREEN = "#10b981"
ACCENT_PURPLE = "#8b5cf6"
ACCENT_RED = "#ef4444"


# ============================================================================
# 1. MAIN DASHBOARD WINDOW
# ============================================================================

class NeuroRhythmDashboard(QMainWindow):
    """Main PyQt5 dashboard for Neuro-Rhythm AI"""

    def __init__(self):
        super().__init__()

        # Initialize modules
        self.initialize_modules()

        # Setup UI
        self.setup_ui()

        # Setup timers
        self.setup_timers()

        # Current state
        self.current_brain_state = "unknown"
        self.is_vision_active = False
        self.is_audio_playing = False
        self.is_rl_learning = False

        print("✅ Neuro-Rhythm AI Dashboard Initialized!")

    def initialize_modules(self):
        """Initialize all AI modules"""
        print("🧠 Initializing AI Modules...")

        # Vision module
        self.vision_module = None
        if VISION_AVAILABLE:
            try:
                self.vision_module = NeuroVision()
                print("   ✅ Vision module loaded")
            except Exception as e:
                print(f"   ❌ Vision module error: {e}")

        # RL module
        self.rl_agent = None
        if RL_AVAILABLE:
            try:
                self.rl_agent = NeuroRLAgent()
                if not self.rl_agent.load_model():
                    print("   ⚠️ No pre-trained RL model found")
                else:
                    print("   ✅ RL agent loaded")
            except Exception as e:
                print(f"   ❌ RL module error: {e}")

        # Audio module
        self.audio_therapist = None
        if AUDIO_AVAILABLE:
            try:
                self.audio_therapist = AudioTherapist(auto_start=False)
                print("   ✅ Audio therapist loaded")
            except Exception as e:
                print(f"   ❌ Audio module error: {e}")

        # Audio mapper for manual mode
        self.audio_mapper = CognitiveAudioMapper() if AUDIO_AVAILABLE else None

    def setup_ui(self):
        """Setup the PyQt5 GUI"""
        self.setWindowTitle("🧠 Neuro-Rhythm AI - Brainwave Therapy Dashboard")
        self.setGeometry(100, 50, 1600, 900)

        # Set window icon
        self.setWindowIcon(QIcon(self.create_brain_icon()))

        # Central widget
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {DARK_BG};")
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # ================= LEFT PANEL (Vision & Controls) =================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # 1. Webcam Feed Section
        webcam_group = self.create_webcam_group()
        left_panel.addWidget(webcam_group)

        # 2. Vision Metrics Section
        metrics_group = self.create_metrics_group()
        left_panel.addWidget(metrics_group)

        # 3. Quick Controls
        controls_group = self.create_controls_group()
        left_panel.addWidget(controls_group)

        # ================= CENTER PANEL (Brain States & Audio) =================
        center_panel = QVBoxLayout()
        center_panel.setSpacing(10)

        # 4. Brain State Visualization
        brain_group = self.create_brain_group()
        center_panel.addWidget(brain_group)

        # 5. Audio Controls
        audio_group = self.create_audio_group()
        center_panel.addWidget(audio_group)

        # 6. RL Agent Status
        rl_group = self.create_rl_group()
        center_panel.addWidget(rl_group)

        # ================= RIGHT PANEL (Logs & Charts) =================
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # 7. Training Log
        log_group = self.create_log_group()
        right_panel.addWidget(log_group)

        # 8. Focus Trend Chart
        chart_group = self.create_chart_group()
        right_panel.addWidget(chart_group)

        # 9. System Status
        status_group = self.create_status_group()
        right_panel.addWidget(status_group)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 35)  # 35% width
        main_layout.addLayout(center_panel, 35)  # 35% width
        main_layout.addLayout(right_panel, 30)  # 30% width

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Select a brain state or enable auto mode")

    def setup_timers(self):
        """Setup QTimers for updates"""
        # Vision update timer (10 FPS)
        self.vision_timer = QTimer()
        self.vision_timer.timeout.connect(self.update_vision)
        self.vision_timer.setInterval(100)  # 100ms = 10 FPS

        # Metrics update timer (1 FPS)
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.setInterval(1000)  # 1 second

        # Chart update timer (2 FPS)
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self.update_charts)
        self.chart_timer.setInterval(500)  # 500ms

    # ============================================================================
    # 2. UI COMPONENT CREATION
    # ============================================================================

    def create_webcam_group(self):
        """Create webcam feed display"""
        group = QGroupBox("🎥 Webcam Feed")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {ACCENT_BLUE};
                font-weight: bold;
                font-size: 14px;
                border: 2px solid {ACCENT_BLUE};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)

        layout = QVBoxLayout()

        # Webcam display label
        self.webcam_label = QLabel()
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setMaximumSize(640, 480)
        self.webcam_label.setStyleSheet(f"""
            border: 2px solid {DARK_PANEL};
            border-radius: 5px;
            background-color: #000000;
        """)
        self.webcam_label.setAlignment(Qt.AlignCenter)

        # Placeholder text if no webcam
        if not VISION_AVAILABLE:
            self.webcam_label.setText("⚠️ Vision Module\nNot Available")
            self.webcam_label.setStyleSheet(f"""
                border: 2px solid {ACCENT_RED};
                border-radius: 5px;
                background-color: #000000;
                color: {ACCENT_RED};
                font-weight: bold;
                font-size: 16px;
            """)

        layout.addWidget(self.webcam_label)

        # Webcam controls
        control_layout = QHBoxLayout()

        self.webcam_button = QPushButton("▶️ Start Webcam")
        self.webcam_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_GREEN};
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #059669;
            }}
        """)
        self.webcam_button.clicked.connect(self.toggle_webcam)

        self.snapshot_button = QPushButton("📸 Take Snapshot")
        self.snapshot_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #0284c7;
            }}
        """)
        self.snapshot_button.clicked.connect(self.take_snapshot)

        control_layout.addWidget(self.webcam_button)
        control_layout.addWidget(self.snapshot_button)
        layout.addLayout(control_layout)

        group.setLayout(layout)
        return group

    def create_metrics_group(self):
        """Create vision metrics display"""
        group = QGroupBox("📊 Cognitive Metrics")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {ACCENT_GREEN};
                font-weight: bold;
                font-size: 14px;
                border: 2px solid {ACCENT_GREEN};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)

        layout = QGridLayout()

        # Focus metric
        self.focus_label = QLabel("Focus: 0%")
        self.focus_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {ACCENT_GREEN};
            padding: 10px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
        """)
        layout.addWidget(QLabel("🎯 Focus:"), 0, 0)
        layout.addWidget(self.focus_label, 0, 1)

        # Stress metric
        self.stress_label = QLabel("Stress: 0%")
        self.stress_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {ACCENT_RED};
            padding: 10px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
        """)
        layout.addWidget(QLabel("😰 Stress:"), 1, 0)
        layout.addWidget(self.stress_label, 1, 1)

        # Drowsiness metric
        self.drowsiness_label = QLabel("Drowsiness: 0%")
        self.drowsiness_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {ACCENT_BLUE};
            padding: 10px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
        """)
        layout.addWidget(QLabel("😴 Drowsiness:"), 2, 0)
        layout.addWidget(self.drowsiness_label, 2, 1)

        # Blink metric
        self.blink_label = QLabel("Blinks: 0")
        self.blink_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: {ACCENT_PURPLE};
            padding: 10px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
        """)
        layout.addWidget(QLabel("👁 Blinks:"), 3, 0)
        layout.addWidget(self.blink_label, 3, 1)

        # Task classification
        self.task_label = QLabel("Task: Unknown")
        self.task_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: #fbbf24;
            padding: 10px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
        """)
        layout.addWidget(QLabel("💼 Task:"), 4, 0)
        layout.addWidget(self.task_label, 4, 1)

        group.setLayout(layout)
        return group

    def create_controls_group(self):
        """Create quick control buttons"""
        group = QGroupBox("⚡ Quick Actions")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {ACCENT_PURPLE};
                font-weight: bold;
                font-size: 14px;
                border: 2px solid {ACCENT_PURPLE};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)

        layout = QVBoxLayout()

        # Auto mode button
        self.auto_button = QPushButton("🤖 Enable Auto Mode")
        self.auto_button.setCheckable(True)
        self.auto_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_PURPLE};
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
                font-size: 14px;
            }}
            QPushButton:checked {{
                background-color: #7c3aed;
            }}
            QPushButton:hover {{
                background-color: #7c3aed;
            }}
        """)
        self.auto_button.clicked.connect(self.toggle_auto_mode)

        # Reset button
        reset_button = QPushButton("🔄 Reset Session")
        reset_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #64748b;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #475569;
            }}
        """)
        reset_button.clicked.connect(self.reset_session)

        # Emergency stop
        stop_button = QPushButton("🛑 Emergency Stop")
        stop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_RED};
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
        """)
        stop_button.clicked.connect(self.emergency_stop)

        layout.addWidget(self.auto_button)
        layout.addWidget(reset_button)
        layout.addWidget(stop_button)
        layout.addStretch()

        group.setLayout(layout)
        return group

    def create_brain_group(self):
        """Create brain state visualization and selection"""
        group = QGroupBox("🧠 Brain State Control")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: #f59e0b;
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #f59e0b;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)

        layout = QVBoxLayout()

        # Current brain state display
        self.brain_state_display = QLabel("Current State: 🟢 Neutral")
        self.brain_state_display.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: #10b981;
            padding: 15px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
            text-align: center;
        """)
        layout.addWidget(self.brain_state_display)

        # Brain state buttons grid
        button_grid = QGridLayout()

        # Brain state buttons with icons and frequencies
        brain_states = [
            ("😴 Deep Sleep", "2.5 Hz Delta", "deep_sleep", "#8b5cf6"),
            ("🧘 Meditation", "6.0 Hz Theta", "meditation", "#3b82f6"),
            ("😌 Relaxed", "10.0 Hz Alpha", "relaxed", "#10b981"),
            ("🎯 Focused", "16.0 Hz Beta", "focused", "#f59e0b"),
            ("🚀 Peak Performance", "40.0 Hz Gamma", "peak_performance", "#ef4444"),
            ("🎨 Creative", "7.83 Hz Theta", "creative_flow", "#ec4899"),
            ("🌊 Stress Relief", "5.5 Hz Blend", "stress_relief", "#06b6d4"),
            ("📚 Memory Boost", "10.5 Hz Alpha", "memory_boost", "#8b5cf6"),
        ]

        for i, (name, freq, preset, color) in enumerate(brain_states):
            btn = QPushButton(f"{name}\n{freq}")
            btn.setProperty("preset", preset)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-weight: bold;
                    padding: 15px;
                    border-radius: 10px;
                    font-size: 12px;
                    text-align: center;
                    min-height: 80px;
                }}
                QPushButton:hover {{
                    background-color: {self.adjust_color(color, -20)};
                }}
                QPushButton:pressed {{
                    background-color: {self.adjust_color(color, -40)};
                }}
            """)
            btn.clicked.connect(self.select_brain_state)
            button_grid.addWidget(btn, i // 2, i % 2)

        layout.addLayout(button_grid)

        # Duration control
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration:"))

        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(60, 1800)  # 1 min to 30 min
        self.duration_slider.setValue(300)  # 5 minutes default
        self.duration_slider.setTickPosition(QSlider.TicksBelow)
        self.duration_slider.setTickInterval(300)

        self.duration_label = QLabel("5:00")
        self.duration_slider.valueChanged.connect(self.update_duration_label)

        duration_layout.addWidget(self.duration_slider)
        duration_layout.addWidget(self.duration_label)
        layout.addLayout(duration_layout)

        group.setLayout(layout)
        return group

    def create_audio_group(self):
        """Create audio controls"""
        group = QGroupBox("🎵 Audio Control")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {ACCENT_BLUE};
                font-weight: bold;
                font-size: 14px;
                border: 2px solid {ACCENT_BLUE};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)

        layout = QVBoxLayout()

        # Current audio display
        self.audio_display = QLabel("Audio: Stopped")
        self.audio_display.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {ACCENT_BLUE};
            padding: 15px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
            text-align: center;
        """)
        layout.addWidget(self.audio_display)

        # Audio control buttons
        audio_controls = QHBoxLayout()

        self.play_button = QPushButton("▶️ Play")
        self.play_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_GREEN};
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                font-size: 14px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: #059669;
            }}
        """)
        self.play_button.clicked.connect(self.toggle_audio)

        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_RED};
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                font-size: 14px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
        """)
        self.stop_button.clicked.connect(self.stop_audio)

        audio_controls.addWidget(self.play_button)
        audio_controls.addWidget(self.stop_button)
        layout.addLayout(audio_controls)

        # Volume control
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("🔊 Volume:"))

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.adjust_volume)

        self.volume_label = QLabel("50%")

        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_label)
        layout.addLayout(volume_layout)

        # Audio test buttons
        test_layout = QHBoxLayout()

        test_all_btn = QPushButton("🎧 Test All")
        test_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_PURPLE};
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #7c3aed;
            }}
        """)
        test_all_btn.clicked.connect(self.test_all_audio)

        save_audio_btn = QPushButton("💾 Save Audio")
        save_audio_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #64748b;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #475569;
            }}
        """)
        save_audio_btn.clicked.connect(self.save_audio_file)

        test_layout.addWidget(test_all_btn)
        test_layout.addWidget(save_audio_btn)
        layout.addLayout(test_layout)

        group.setLayout(layout)
        return group

    def create_rl_group(self):
        """Create RL agent controls"""
        group = QGroupBox("🤖 Reinforcement Learning")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: #ec4899;
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #ec4899;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)

        layout = QVBoxLayout()

        # RL status
        self.rl_status_label = QLabel("RL: Not Active")
        self.rl_status_label.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            color: #ec4899;
            padding: 10px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
            text-align: center;
        """)
        layout.addWidget(self.rl_status_label)

        # RL recommendation
        self.rl_recommendation_label = QLabel("Recommendation: None")
        self.rl_recommendation_label.setStyleSheet(f"""
            font-size: 14px;
            color: {DARK_TEXT};
            padding: 10px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
        """)
        layout.addWidget(self.rl_recommendation_label)

        # RL confidence
        self.rl_confidence_label = QLabel("Confidence: 0%")
        self.rl_confidence_label.setStyleSheet(f"""
            font-size: 14px;
            color: {DARK_TEXT};
            padding: 10px;
            border-radius: 10px;
            background-color: {DARK_PANEL};
        """)
        layout.addWidget(self.rl_confidence_label)

        # RL controls
        rl_controls = QHBoxLayout()

        self.train_rl_btn = QPushButton("🎓 Train RL")
        self.train_rl_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #f59e0b;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #d97706;
            }}
        """)
        self.train_rl_btn.clicked.connect(self.train_rl_agent)

        self.load_rl_btn = QPushButton("📂 Load Model")
        self.load_rl_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #0284c7;
            }}
        """)
        self.load_rl_btn.clicked.connect(self.load_rl_model)

        rl_controls.addWidget(self.train_rl_btn)
        rl_controls.addWidget(self.load_rl_btn)
        layout.addLayout(rl_controls)

        # RL learning toggle
        self.rl_learning_toggle = QCheckBox("Enable RL Learning")
        self.rl_learning_toggle.setStyleSheet(f"""
            QCheckBox {{
                color: {DARK_TEXT};
                font-size: 14px;
                padding: 10px;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
            }}
        """)
        self.rl_learning_toggle.stateChanged.connect(self.toggle_rl_learning)
        layout.addWidget(self.rl_learning_toggle)

        group.setLayout(layout)
        return group

    def create_log_group(self):
        """Create training and session log"""
        group = QGroupBox("📝 Session Log")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: #64748b;
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #64748b;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)

        layout = QVBoxLayout()

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {DARK_PANEL};
                color: {DARK_TEXT};
                font-family: 'Consolas', 'Monospace';
                font-size: 12px;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 10px;
            }}
        """)
        self.log_text.setMaximumHeight(200)

        # Add welcome message
        self.log_message("=" * 50)
        self.log_message("🧠 NEURO-RHYTHM AI SESSION STARTED")
        self.log_message("=" * 50)
        self.log_message(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message("System initialized successfully")

        layout.addWidget(self.log_text)

        # Log controls
        log_controls = QHBoxLayout()

        clear_log_btn = QPushButton("🗑️ Clear Log")
        clear_log_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #64748b;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #475569;
            }}
        """)
        clear_log_btn.clicked.connect(self.clear_log)

        save_log_btn = QPushButton("💾 Save Log")
        save_log_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #0284c7;
            }}
        """)
        save_log_btn.clicked.connect(self.save_log)

        log_controls.addWidget(clear_log_btn)
        log_controls.addWidget(save_log_btn)
        layout.addLayout(log_controls)

        group.setLayout(layout)
        return group

    def create_chart_group(self):
        """Create focus trend chart"""
        group = QGroupBox("📈 Focus Trend")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {ACCENT_GREEN};
                font-weight: bold;
                font-size: 14px;
                border: 2px solid {ACCENT_GREEN};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)

        layout = QVBoxLayout()

        # Create PyQtGraph plot
        self.focus_plot = pg.PlotWidget()
        self.focus_plot.setBackground(DARK_PANEL)
        self.focus_plot.setLabel('left', 'Focus (%)', color=DARK_TEXT)
        self.focus_plot.setLabel('bottom', 'Time', color=DARK_TEXT)
        self.focus_plot.setTitle('Focus Level Over Time', color=DARK_TEXT, size='14pt')
        self.focus_plot.showGrid(x=True, y=True, alpha=0.3)
        self.focus_plot.setYRange(0, 100)

        # Plot curve
        self.focus_curve = self.focus_plot.plot(pen=pg.mkPen(color=ACCENT_GREEN, width=3))

        # Data storage
        self.focus_data = []
        self.time_data = []

        layout.addWidget(self.focus_plot)
        group.setLayout(layout)
        return group

    def create_status_group(self):
        """Create system status panel"""
        group = QGroupBox("⚙️ System Status")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: #64748b;
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #64748b;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)

        layout = QVBoxLayout()

        # Status indicators
        status_grid = QGridLayout()

        # Vision status
        self.vision_status = QLabel("● Vision: Ready")
        self.vision_status.setStyleSheet(f"color: {ACCENT_GREEN}; font-weight: bold;")
        status_grid.addWidget(self.vision_status, 0, 0)

        # Audio status
        self.audio_status = QLabel("● Audio: Ready")
        self.audio_status.setStyleSheet(f"color: {ACCENT_BLUE}; font-weight: bold;")
        status_grid.addWidget(self.audio_status, 1, 0)

        # RL status
        self.rl_system_status = QLabel("● RL: Ready")
        self.rl_system_status.setStyleSheet(f"color: #ec4899; font-weight: bold;")
        status_grid.addWidget(self.rl_system_status, 2, 0)

        # Session time
        self.session_time_label = QLabel("🕐 Session: 00:00:00")
        self.session_time_label.setStyleSheet(f"color: {DARK_TEXT}; font-weight: bold;")
        status_grid.addWidget(self.session_time_label, 3, 0)

        layout.addLayout(status_grid)

        # Session stats
        stats_group = QGroupBox("📊 Session Statistics")
        stats_group.setStyleSheet(f"""
            QGroupBox {{
                color: #94a3b8;
                font-size: 12px;
                border: 1px solid #475569;
                border-radius: 5px;
                margin-top: 5px;
            }}
        """)

        stats_layout = QVBoxLayout()

        self.stats_audio_time = QLabel("🎵 Audio Time: 0s")
        self.stats_focus_avg = QLabel("🎯 Avg Focus: 0%")
        self.stats_state_changes = QLabel("🔄 State Changes: 0")

        for label in [self.stats_audio_time, self.stats_focus_avg, self.stats_state_changes]:
            label.setStyleSheet(f"color: {DARK_TEXT}; font-size: 11px;")
            stats_layout.addWidget(label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # System info
        info_label = QLabel("Neuro-Rhythm AI v1.0\n© 2024 AI Therapist")
        info_label.setStyleSheet(f"""
            color: #94a3b8;
            font-size: 10px;
            text-align: center;
            padding: 10px;
            border-top: 1px solid #475569;
            margin-top: 10px;
        """)
        layout.addWidget(info_label)

        group.setLayout(layout)
        return group

    # ============================================================================
    # 3. HELPER FUNCTIONS
    # ============================================================================

    def create_brain_icon(self):
        """Create a simple brain icon for the window"""
        from PyQt5.QtGui import QPainter, QBrush, QPen, QColor
        from PyQt5.QtCore import Qt

        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw brain icon
        painter.setBrush(QBrush(QColor(ACCENT_PURPLE)))
        painter.setPen(QPen(Qt.NoPen))

        # Draw brain-like shape
        painter.drawEllipse(20, 15, 24, 30)  # Main brain
        painter.drawEllipse(15, 20, 10, 15)  # Left lobe
        painter.drawEllipse(39, 20, 10, 15)  # Right lobe

        painter.end()
        return pixmap

    def adjust_color(self, hex_color, amount):
        """Adjust color brightness"""
        from PyQt5.QtGui import QColor

        color = QColor(hex_color)
        h, s, v, a = color.getHsv()
        v = max(0, min(255, v + amount))
        return QColor.fromHsv(h, s, v, a).name()

    def log_message(self, message):
        """Add message to log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.ensureCursorVisible()

    # ============================================================================
    # 4. BUTTON HANDLERS
    # ============================================================================

    def toggle_webcam(self):
        """Toggle webcam feed"""
        if not VISION_AVAILABLE or not self.vision_module:
            self.log_message("❌ Vision module not available")
            return

        if not self.is_vision_active:
            # Start webcam
            self.vision_timer.start()
            self.metrics_timer.start()
            self.chart_timer.start()
            self.is_vision_active = True
            self.webcam_button.setText("⏸️ Stop Webcam")
            self.log_message("🎥 Webcam started")
        else:
            # Stop webcam
            self.vision_timer.stop()
            self.metrics_timer.stop()
            self.chart_timer.stop()
            self.is_vision_active = False
            self.webcam_button.setText("▶️ Start Webcam")
            self.webcam_label.clear()
            self.log_message("🎥 Webcam stopped")

    def take_snapshot(self):
        """Take snapshot of current frame"""
        if not self.is_vision_active:
            self.log_message("❌ Webcam not active")
            return

        # Save snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.png"

        # Get current pixmap from label
        pixmap = self.webcam_label.pixmap()
        if pixmap:
            pixmap.save(filename)
            self.log_message(f"📸 Snapshot saved: {filename}")

    def select_brain_state(self):
        """Handle brain state button click"""
        sender = self.sender()
        preset_name = sender.property("preset")

        if not AUDIO_AVAILABLE or not self.audio_therapist:
            self.log_message("❌ Audio module not available")
            return

        # Update brain state display
        self.current_brain_state = preset_name

        # Get preset info
        if preset_name in AUDIO_PRESETS:
            preset = AUDIO_PRESETS[preset_name]
            self.brain_state_display.setText(
                f"Current State: {preset.get('icon', '🧠')} "
                f"{preset_name.replace('_', ' ').title()}\n"
                f"Frequency: {preset.get('beat_freq', 0)} Hz"
            )

        # Log the selection
        self.log_message(f"🧠 Brain state selected: {preset_name}")

        # If auto mode is off, play the selected preset
        if not self.auto_button.isChecked():
            self.play_selected_audio()

    def update_duration_label(self, value):
        """Update duration label when slider changes"""
        minutes = value // 60
        seconds = value % 60
        self.duration_label.setText(f"{minutes}:{seconds:02d}")

    def toggle_audio(self):
        """Toggle audio playback"""
        if not AUDIO_AVAILABLE or not self.audio_therapist:
            self.log_message("❌ Audio module not available")
            return

        if not self.is_audio_playing:
            # Start audio
            self.play_selected_audio()
        else:
            # Stop audio
            self.stop_audio()

    def play_selected_audio(self):
        """Play the selected brain state audio"""
        if not self.current_brain_state or self.current_brain_state == "unknown":
            self.log_message("❌ No brain state selected")
            return

        duration = self.duration_slider.value()

        try:
            self.audio_therapist.play_preset(self.current_brain_state, duration_seconds=duration)
            self.is_audio_playing = True
            self.play_button.setText("⏸️ Pause")
            self.audio_display.setText(f"Playing: {self.current_brain_state.replace('_', ' ').title()}")
            self.log_message(f"🎵 Playing audio: {self.current_brain_state} ({duration}s)")
        except Exception as e:
            self.log_message(f"❌ Audio error: {e}")

    def stop_audio(self):
        """Stop audio playback"""
        if AUDIO_AVAILABLE and self.audio_therapist:
            self.audio_therapist.stop()
            self.is_audio_playing = False
            self.play_button.setText("▶️ Play")
            self.audio_display.setText("Audio: Stopped")
            self.log_message("🎵 Audio stopped")

    def adjust_volume(self, value):
        """Adjust audio volume"""
        self.volume_label.setText(f"{value}%")
        if AUDIO_AVAILABLE and self.audio_therapist:
            volume = value / 100.0
            self.audio_therapist.adjust_volume(volume)

    def test_all_audio(self):
        """Test all audio presets"""
        if not AUDIO_AVAILABLE or not self.audio_therapist:
            self.log_message("❌ Audio module not available")
            return

        self.log_message("🎧 Starting audio test sequence...")

        # Run in separate thread to avoid blocking GUI
        def test_sequence():
            test_presets = [
                'deep_sleep', 'meditation', 'relaxed',
                'focused', 'peak_performance', 'creative_flow'
            ]

            for preset in test_presets:
                QMetaObject.invokeMethod(self, "log_message",
                                         Qt.QueuedConnection,
                                         Q_ARG(str, f"🎵 Testing: {preset}"))

                # Update GUI from main thread
                QMetaObject.invokeMethod(self, "update_test_display",
                                         Qt.QueuedConnection,
                                         Q_ARG(str, preset))

                # Play preset for 5 seconds
                self.audio_therapist.play_preset(preset, duration_seconds=5)
                time.sleep(7)  # 5s play + 2s buffer

            QMetaObject.invokeMethod(self, "log_message",
                                     Qt.QueuedConnection,
                                     Q_ARG(str, "✅ Audio test complete"))

            QMetaObject.invokeMethod(self, "update_test_display",
                                     Qt.QueuedConnection,
                                     Q_ARG(str, "complete"))

        threading.Thread(target=test_sequence, daemon=True).start()

    def update_test_display(self, preset):
        """Update display during audio test"""
        if preset == "complete":
            self.audio_display.setText("Audio Test Complete")
        else:
            self.audio_display.setText(f"Testing: {preset}")

    def save_audio_file(self):
        """Save current audio to file"""
        if not self.current_brain_state or self.current_brain_state == "unknown":
            self.log_message("❌ No brain state selected")
            return

        if AUDIO_AVAILABLE and self.audio_therapist:
            filename = self.audio_therapist.save_audio_to_file(
                self.current_brain_state,
                duration_seconds=10
            )
            if filename:
                self.log_message(f"💾 Audio saved: {filename}")

    def toggle_auto_mode(self):
        """Toggle auto mode (RL-driven)"""
        if self.auto_button.isChecked():
            self.log_message("🤖 Auto mode enabled")
            # Start RL updates
        else:
            self.log_message("🤖 Auto mode disabled")

    def reset_session(self):
        """Reset current session"""
        self.stop_audio()

        # Reset data
        self.focus_data = []
        self.time_data = []
        self.focus_curve.clear()

        # Reset labels
        self.focus_label.setText("Focus: 0%")
        self.stress_label.setText("Stress: 0%")
        self.drowsiness_label.setText("Drowsiness: 0%")
        self.blink_label.setText("Blinks: 0")
        self.task_label.setText("Task: Unknown")

        self.log_message("🔄 Session reset")

    def emergency_stop(self):
        """Emergency stop all systems"""
        self.log_message("🛑 EMERGENCY STOP - Stopping all systems")

        # Stop timers
        self.vision_timer.stop()
        self.metrics_timer.stop()
        self.chart_timer.stop()

        # Stop audio
        self.stop_audio()

        # Reset webcam
        self.is_vision_active = False
        self.webcam_button.setText("▶️ Start Webcam")

        self.log_message("✅ All systems stopped")

    def train_rl_agent(self):
        """Train RL agent"""
        if not RL_AVAILABLE or not self.rl_agent:
            self.log_message("❌ RL module not available")
            return

        self.log_message("🎓 Starting RL training...")

        # Run training in separate thread
        def train_thread():
            try:
                # Train for 50 episodes
                rewards, losses = self.rl_agent.train_simulation(episodes=50)

                # Update GUI
                QMetaObject.invokeMethod(self, "log_message",
                                         Qt.QueuedConnection,
                                         Q_ARG(str,
                                               f"✅ RL training complete. Avg reward: {np.mean(rewards[-10:]):.2f}"))

                QMetaObject.invokeMethod(self, "rl_status_label",
                                         Qt.QueuedConnection,
                                         Q_ARG(str, f"RL: Trained (Reward: {np.mean(rewards[-10:]):.2f})"))

            except Exception as e:
                QMetaObject.invokeMethod(self, "log_message",
                                         Qt.QueuedConnection,
                                         Q_ARG(str, f"❌ RL training error: {e}"))

        threading.Thread(target=train_thread, daemon=True).start()

    def load_rl_model(self):
        """Load RL model"""
        if not RL_AVAILABLE or not self.rl_agent:
            self.log_message("❌ RL module not available")
            return

        if self.rl_agent.load_model():
            self.log_message("✅ RL model loaded successfully")
            self.rl_status_label.setText("RL: Model Loaded")
        else:
            self.log_message("❌ Failed to load RL model")

    def toggle_rl_learning(self, state):
        """Toggle RL learning mode"""
        self.is_rl_learning = state == Qt.Checked
        status = "enabled" if self.is_rl_learning else "disabled"
        self.log_message(f"🤖 RL learning {status}")

    def clear_log(self):
        """Clear log messages"""
        self.log_text.clear()
        self.log_message("Log cleared")

    def save_log(self):
        """Save log to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_log_{timestamp}.txt"

        try:
            with open(filename, 'w') as f:
                f.write(self.log_text.toPlainText())
            self.log_message(f"📝 Log saved: {filename}")
        except Exception as e:
            self.log_message(f"❌ Error saving log: {e}")

    # ============================================================================
    # 5. UPDATE FUNCTIONS
    # ============================================================================

    def update_vision(self):
        """Update webcam feed"""
        if not VISION_AVAILABLE or not self.vision_module or not self.is_vision_active:
            return

        try:
            # Get frame from vision module
            # Note: This depends on your vision.py implementation
            # You might need to adapt this based on how your vision module works

            # For now, simulate or use actual camera
            # Replace this with actual vision module call
            pass

        except Exception as e:
            print(f"Vision update error: {e}")

    def update_metrics(self):
        """Update cognitive metrics"""
        # Simulate metrics for now - replace with actual vision module data
        import random

        focus = random.randint(40, 90)
        stress = random.randint(10, 70)
        drowsiness = random.randint(5, 50)
        blinks = random.randint(0, 10)
        tasks = ["Focus", "Creative", "Reading", "Browsing", "Unknown"]
        task = random.choice(tasks)

        # Update labels
        self.focus_label.setText(f"Focus: {focus}%")
        self.stress_label.setText(f"Stress: {stress}%")
        self.drowsiness_label.setText(f"Drowsiness: {drowsiness}%")
        self.blink_label.setText(f"Blinks: {blinks}")
        self.task_label.setText(f"Task: {task}")

        # Store for chart
        self.focus_data.append(focus)
        self.time_data.append(len(self.focus_data))

        # If auto mode is on and RL is available, get recommendation
        if (self.auto_button.isChecked() and RL_AVAILABLE and self.rl_agent and
                self.is_rl_learning):

            # Create state vector
            state = {
                'focus': focus,
                'stress': stress,
                'drowsiness': drowsiness,
                'task': task.lower()
            }

            try:
                # Get RL recommendation
                action_idx, preset = self.rl_agent.get_audio_recommendation(state)

                # Update RL display
                self.rl_recommendation_label.setText(
                    f"Recommendation: {preset.get('name', 'Unknown')}"
                )
                self.rl_confidence_label.setText(
                    f"Confidence: {preset.get('confidence', 0) * 100:.1f}%"
                )

                # Apply recommendation
                if preset.get('name') in AUDIO_PRESETS:
                    self.current_brain_state = preset['name']
                    if not self.is_audio_playing:
                        self.play_selected_audio()

            except Exception as e:
                print(f"RL update error: {e}")

    def update_charts(self):
        """Update focus trend chart"""
        if len(self.focus_data) > 0:
            # Keep only last 100 points
            if len(self.focus_data) > 100:
                self.focus_data = self.focus_data[-100:]
                self.time_data = self.time_data[-100:]

            # Update plot
            self.focus_curve.setData(self.time_data, self.focus_data)

    def update_status(self):
        """Update system status"""
        # Update session time
        if hasattr(self, 'session_start_time'):
            elapsed = time.time() - self.session_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.session_time_label.setText(f"🕐 Session: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def closeEvent(self, event):
        """Clean up when closing window"""
        self.log_message("=" * 50)
        self.log_message("👋 Session ended")
        self.log_message("=" * 50)

        # Stop all systems
        self.emergency_stop()

        # Close modules
        if AUDIO_AVAILABLE and self.audio_therapist:
            self.audio_therapist.close()

        event.accept()


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set application icon and name
    app.setApplicationName("Neuro-Rhythm AI")
    app.setApplicationDisplayName("🧠 Neuro-Rhythm AI - Brainwave Therapy")

    # Create and show main window
    window = NeuroRhythmDashboard()
    window.show()

    # Start session timer
    window.session_start_time = time.time()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
