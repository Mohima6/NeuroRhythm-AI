"""
NEURO-RHYTHM AI - Brainwave Audio Engine
File: vision_engine/audio.py
Description: Real-time binaural beats & isochronic tones generator
"""

import numpy as np
import pyaudio
import threading
import time
import math
import wave
import os
from datetime import datetime
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. AUDIO ENGINE CONSTANTS
# ============================================================================

# Audio parameters
SAMPLE_RATE = 44100  # CD quality
CHUNK_SIZE = 1024  # Audio buffer size
DURATION = 300  # Default duration in seconds (5 minutes)

# Brainwave frequency ranges (Hz)
BRAINWAVE_RANGES = {
    'DELTA': (0.5, 4.0),  # Deep sleep, healing
    'THETA': (4.0, 8.0),  # Meditation, creativity
    'ALPHA': (8.0, 13.0),  # Relaxed focus
    'BETA': (13.0, 30.0),  # Active thinking
    'GAMMA': (30.0, 100.0)  # Peak performance
}

# ============================================================================
# 2. AUDIO PRESETS WITH REAL FREQUENCIES
# ============================================================================

AUDIO_PRESETS = {
    "deep_sleep": {
        "type": "binaural",
        "base_freq": 200,  # Carrier frequency
        "beat_freq": 2.5,  # Delta waves for sleep
        "volume": 0.3,
        "pan": 0.0,  # Center pan
        "description": "Deep sleep & healing",
        "color": "#5D3FD3",
        "icon": "😴"
    },
    "meditation": {
        "type": "binaural",
        "base_freq": 200,
        "beat_freq": 6.0,  # Theta waves for meditation
        "volume": 0.4,
        "pan": -0.3,  # Slight left pan
        "description": "Deep meditation & intuition",
        "color": "#1E90FF",
        "icon": "🧘"
    },
    "relaxed": {
        "type": "binaural",
        "base_freq": 210,
        "beat_freq": 10.0,  # Alpha waves for relaxation
        "volume": 0.5,
        "pan": 0.0,
        "description": "Calm & relaxed focus",
        "color": "#32CD32",
        "icon": "😌"
    },
    "focused": {
        "type": "isochronic",
        "base_freq": 500,  # Higher carrier for focus
        "beat_freq": 16.0,  # Beta waves for focus
        "volume": 0.6,
        "pan": 0.2,  # Slight right pan
        "pulse_width": 0.5,  # 50% duty cycle
        "description": "Active thinking & problem solving",
        "color": "#FFD700",
        "icon": "🎯"
    },
    "peak_performance": {
        "type": "isochronic",
        "base_freq": 600,
        "beat_freq": 40.0,  # Gamma waves for peak performance
        "volume": 0.5,
        "pan": 0.0,
        "pulse_width": 0.3,  # 30% duty cycle
        "description": "Peak concentration & cognition",
        "color": "#FF4500",
        "icon": "🚀"
    },
    "creative_flow": {
        "type": "binaural",
        "base_freq": 220,
        "beat_freq": 7.83,  # Schumann resonance for creativity
        "volume": 0.45,
        "pan": 0.0,
        "description": "Creative thinking & flow state",
        "color": "#9B30FF",
        "icon": "🎨"
    },
    "stress_relief": {
        "type": "binaural",
        "base_freq": 180,
        "beat_freq": 5.5,  # Theta-Alpha border for stress relief
        "volume": 0.35,
        "pan": 0.3,  # Slight right pan
        "description": "Stress reduction & anxiety relief",
        "color": "#00CED1",
        "icon": "🌊"
    },
    "memory_boost": {
        "type": "isochronic",
        "base_freq": 400,
        "beat_freq": 10.5,  # Alpha for memory
        "volume": 0.55,
        "pan": -0.2,
        "pulse_width": 0.4,
        "description": "Memory enhancement & learning",
        "color": "#FF69B4",
        "icon": "🧠"
    },
    "silence": {
        "type": "silence",
        "volume": 0.0,
        "description": "No audio - system reset",
        "color": "#708090",
        "icon": "🔇"
    }
}


# ============================================================================
# 3. AUDIO GENERATION FUNCTIONS
# ============================================================================

class AudioGenerator:
    """Core audio signal generation"""

    @staticmethod
    def generate_sine_wave(frequency, duration, sample_rate=SAMPLE_RATE, volume=0.5):
        """Generate a sine wave tone"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        return wave * volume

    @staticmethod
    def generate_binaural_beats(base_freq, beat_freq, duration, volume=0.5):
        """
        Generate binaural beats:
        - Left ear: base_freq - beat_freq/2
        - Right ear: base_freq + beat_freq/2
        Brain perceives: beat_freq difference
        """
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

        left_freq = base_freq - beat_freq / 2
        right_freq = base_freq + beat_freq / 2

        left_channel = np.sin(2 * np.pi * left_freq * t)
        right_channel = np.sin(2 * np.pi * right_freq * t)

        # Apply volume
        left_channel *= volume
        right_channel *= volume

        # Create stereo array
        stereo = np.column_stack((left_channel, right_channel))
        return stereo.astype(np.float32)

    @staticmethod
    def generate_isochronic_tones(carrier_freq, beat_freq, duration, volume=0.5, pulse_width=0.5):
        """
        Generate isochronic tones (pulsed sine waves)
        - On/Off pulses at beat_freq rate
        - carrier_freq determines the tone pitch
        """
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

        # Generate carrier wave
        carrier = np.sin(2 * np.pi * carrier_freq * t)

        # Generate pulse envelope at beat_freq
        pulse_rate = beat_freq  # Hz
        pulse_samples = int(SAMPLE_RATE / pulse_rate)
        pulse_on_samples = int(pulse_samples * pulse_width)

        # Create envelope
        envelope = np.zeros_like(t)
        for i in range(0, len(t), pulse_samples):
            envelope[i:i + pulse_on_samples] = 1.0

        # Apply envelope to carrier
        modulated = carrier * envelope

        # Create stereo (same in both channels)
        stereo = np.column_stack((modulated, modulated))

        return (stereo * volume).astype(np.float32)

    @staticmethod
    def generate_nature_sounds(duration, volume=0.4):
        """Generate nature-like sounds (rain, ocean) for relaxation"""
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

        # Brown noise (more natural than white noise)
        brown = np.cumsum(np.random.randn(len(t))) - 0.5
        brown = brown / np.max(np.abs(brown))

        # Add some gentle modulation
        mod_freq = 0.1  # Very slow modulation
        envelope = 0.5 + 0.3 * np.sin(2 * np.pi * mod_freq * t)

        result = brown * envelope

        # Stereo with slight variation
        left = result * 0.8
        right = result * 1.0
        stereo = np.column_stack((left, right))

        return (stereo * volume).astype(np.float32)


# ============================================================================
# 4. MAIN AUDIO THERAPIST CLASS
# ============================================================================

class AudioTherapist:
    """
    Main audio engine for Neuro-Rhythm AI
    Generates real-time brainwave entrainment audio
    """

    def __init__(self, auto_start=True):
        print("🎵 Initializing Audio Therapist...")

        # PyAudio instance
        self.p = pyaudio.PyAudio()

        # Audio stream
        self.stream = None
        self.is_playing = False
        self.current_preset = None
        self.current_frequencies = None

        # Threading
        self.audio_thread = None
        self.stop_signal = threading.Event()

        # Audio history
        self.audio_history = deque(maxlen=100)
        self.start_time = None

        # Volume control
        self.master_volume = 0.7

        # Check audio devices
        self._list_audio_devices()

        print("✅ Audio Therapist ready!")
        print("   Available presets:")
        for name, preset in AUDIO_PRESETS.items():
            if name != "silence":
                print(f"   • {preset['icon']} {name.replace('_', ' ').title():20} - {preset['beat_freq']} Hz")

    def _list_audio_devices(self):
        """List available audio devices"""
        print("\n🔊 Audio Devices Found:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:  # Output devices only
                print(f"   [{i}] {info['name']} (Channels: {info['maxOutputChannels']})")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback function for real-time audio generation
        This is called repeatedly to fill audio buffer
        """
        if self.stop_signal.is_set():
            return (np.zeros((frame_count, 2), dtype=np.float32).tobytes(),
                    pyaudio.paComplete)

        # Generate audio based on current preset
        if self.current_preset and self.current_preset != "silence":
            preset = AUDIO_PRESETS[self.current_preset]

            # Calculate time for this chunk
            chunk_duration = frame_count / SAMPLE_RATE

            if preset['type'] == 'binaural':
                audio_data = AudioGenerator.generate_binaural_beats(
                    preset['base_freq'],
                    preset['beat_freq'],
                    chunk_duration,
                    preset['volume'] * self.master_volume
                )
            elif preset['type'] == 'isochronic':
                audio_data = AudioGenerator.generate_isochronic_tones(
                    preset['base_freq'],
                    preset['beat_freq'],
                    chunk_duration,
                    preset['volume'] * self.master_volume,
                    preset.get('pulse_width', 0.5)
                )
            else:
                # Fallback to sine wave
                audio_data = AudioGenerator.generate_sine_wave(
                    preset['base_freq'],
                    chunk_duration,
                    SAMPLE_RATE,
                    preset['volume'] * self.master_volume
                )
                audio_data = np.column_stack((audio_data, audio_data))

            # Apply panning if specified
            if 'pan' in preset and preset['pan'] != 0:
                pan = preset['pan']
                left_gain = 1.0 - max(0, pan)
                right_gain = 1.0 + min(0, pan)
                audio_data[:, 0] *= left_gain
                audio_data[:, 1] *= right_gain

            # Ensure correct shape
            if audio_data.shape[0] < frame_count:
                # Pad if needed
                padding = np.zeros((frame_count - audio_data.shape[0], 2), dtype=np.float32)
                audio_data = np.vstack((audio_data, padding))
            elif audio_data.shape[0] > frame_count:
                # Trim if needed
                audio_data = audio_data[:frame_count]

            return (audio_data.tobytes(), pyaudio.paContinue)

        # Return silence if no preset
        return (np.zeros((frame_count, 2), dtype=np.float32).tobytes(),
                pyaudio.paContinue)

    def play_preset(self, preset_name="relaxed", duration_seconds=300):
        """
        Play a brainwave audio preset
        preset_name: one of AUDIO_PRESETS keys
        duration_seconds: how long to play (0 = infinite)
        """
        if preset_name not in AUDIO_PRESETS:
            print(f"⚠️ Unknown preset: {preset_name}. Using 'relaxed'.")
            preset_name = "relaxed"

        preset = AUDIO_PRESETS[preset_name]

        # Stop any current playback
        self.stop()
        time.sleep(0.1)  # Brief pause

        if preset_name == "silence":
            print(f"{preset['icon']} Audio stopped")
            self.current_preset = None
            return

        # Update state
        self.current_preset = preset_name
        self.current_frequencies = {
            'beat': preset.get('beat_freq', 0),
            'base': preset.get('base_freq', 0),
            'type': preset['type']
        }

        # Start audio stream
        print(f"\n🎧 Starting {preset['icon']} {preset_name.replace('_', ' ').title()}")
        print(f"   Frequency: {preset.get('beat_freq', 0):.1f} Hz ({preset['type']})")
        print(f"   Volume: {preset['volume']:.1f}")
        print(f"   Duration: {duration_seconds}s")
        print(f"   Description: {preset['description']}")

        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=2,  # Stereo
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._audio_callback
        )

        self.is_playing = True
        self.stop_signal.clear()
        self.start_time = time.time()

        # Log
        self.audio_history.append({
            'preset': preset_name,
            'frequency': preset.get('beat_freq', 0),
            'start_time': self.start_time,
            'duration': duration_seconds,
            'type': preset['type']
        })

        # Start duration timer if specified
        if duration_seconds > 0:
            def stop_after_duration():
                time.sleep(duration_seconds)
                if self.is_playing and self.current_preset == preset_name:
                    print(f"\n⏰ Duration reached. Stopping {preset_name}...")
                    self.stop()

            timer_thread = threading.Thread(target=stop_after_duration, daemon=True)
            timer_thread.start()

        # Start the stream
        self.stream.start_stream()
        print("   ▶️ Audio playing... (Press Ctrl+C to stop)")

    def stop(self):
        """Stop audio playback"""
        if self.stream and self.is_playing:
            print("\n⏹️ Stopping audio playback...")
            self.stop_signal.set()
            time.sleep(0.1)  # Let callback finish

            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

            self.is_playing = False

            if self.current_preset and self.start_time:
                duration = time.time() - self.start_time
                print(f"   Played for {duration:.1f} seconds")

            self.current_preset = None
            self.current_frequencies = None

    def fade_out(self, fade_duration=3.0):
        """Gradually fade out audio"""
        if not self.is_playing:
            return

        original_volume = self.master_volume
        steps = int(fade_duration * 10)  # 10 steps per second

        print(f"🔉 Fading out over {fade_duration}s...")

        for i in range(steps):
            if not self.is_playing:
                break

            # Linear fade
            self.master_volume = original_volume * (1.0 - i / steps)
            time.sleep(0.1)

        self.stop()
        self.master_volume = original_volume  # Reset

    def adjust_volume(self, volume_level):
        """Adjust master volume (0.0 to 1.0)"""
        self.master_volume = max(0.0, min(1.0, volume_level))
        print(f"🔊 Volume set to {self.master_volume:.1f}")

    def get_current_audio_info(self):
        """Get information about currently playing audio"""
        if not self.is_playing or not self.current_preset:
            return None

        preset = AUDIO_PRESETS[self.current_preset]

        return {
            'preset': self.current_preset,
            'name': preset['description'],
            'frequency': preset.get('beat_freq', 0),
            'type': preset['type'],
            'volume': preset['volume'] * self.master_volume,
            'duration': time.time() - self.start_time if self.start_time else 0,
            'icon': preset['icon'],
            'color': preset['color']
        }

    def save_audio_to_file(self, preset_name, duration_seconds=10, filename=None):
        """
        Save generated audio to WAV file for testing
        """
        if preset_name not in AUDIO_PRESETS:
            print(f"⚠️ Unknown preset: {preset_name}")
            return

        preset = AUDIO_PRESETS[preset_name]

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_{preset_name}_{timestamp}.wav"

        print(f"💾 Saving {preset_name} audio to {filename}...")

        # Generate audio
        if preset['type'] == 'binaural':
            audio_data = AudioGenerator.generate_binaural_beats(
                preset['base_freq'],
                preset['beat_freq'],
                duration_seconds,
                preset['volume']
            )
        elif preset['type'] == 'isochronic':
            audio_data = AudioGenerator.generate_isochronic_tones(
                preset['base_freq'],
                preset['beat_freq'],
                duration_seconds,
                preset['volume'],
                preset.get('pulse_width', 0.5)
            )
        else:
            # Mono sine wave
            mono = AudioGenerator.generate_sine_wave(
                preset.get('base_freq', 440),
                duration_seconds,
                SAMPLE_RATE,
                preset['volume']
            )
            audio_data = np.column_stack((mono, mono))

        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Save as WAV file
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 2 bytes = 16 bits
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())

        print(f"✅ Audio saved to {filename} ({duration_seconds}s)")
        return filename

    def play_test_sequence(self, duration_per_preset=10):
        """
        Play a test sequence of all presets
        Great for testing headphones/speakers
        """
        print("\n🎵 Starting Audio Test Sequence")
        print("   This will play each brainwave preset for 10 seconds")
        print("   Make sure your speakers/headphones are working!\n")

        presets_to_test = [
            'deep_sleep', 'meditation', 'relaxed',
            'focused', 'peak_performance', 'creative_flow',
            'stress_relief', 'memory_boost'
        ]

        for preset_name in presets_to_test:
            print(f"\n▶️ Testing: {preset_name.replace('_', ' ').title()}")
            self.play_preset(preset_name, duration_per_preset)
            time.sleep(duration_per_preset + 1)  # Play + buffer

        print("\n✅ Audio test complete!")
        print("   All presets should have played correctly.")
        print("   If you didn't hear anything, check your audio output device.")

    def close(self):
        """Clean up audio resources"""
        print("\n🔇 Closing Audio Therapist...")
        self.stop()
        self.p.terminate()
        print("✅ Audio resources released")


# ============================================================================
# 5. COGNITIVE STATE TO AUDIO MAPPING
# ============================================================================

class CognitiveAudioMapper:
    """
    Maps cognitive states (from vision/RL) to audio presets
    """

    @staticmethod
    def map_state_to_audio(focus, stress, drowsiness, task_type="unknown"):
        """
        Map cognitive metrics to appropriate audio preset
        focus: 0-100% (higher = more focused)
        stress: 0-100% (higher = more stressed)
        drowsiness: 0-100% (higher = more drowsy)
        task_type: string describing current task
        """
        # Normalize to 0-1
        focus_norm = focus / 100.0
        stress_norm = stress / 100.0
        drowsiness_norm = drowsiness / 100.0

        # Decision logic
        if stress_norm > 0.7:
            return "stress_relief"
        elif drowsiness_norm > 0.6:
            return "deep_sleep"
        elif focus_norm < 0.4:
            if task_type in ["coding", "studying", "reading"]:
                return "focused"
            else:
                return "creative_flow"
        elif focus_norm > 0.7:
            if stress_norm < 0.3:
                return "peak_performance"
            else:
                return "focused"
        elif task_type in ["creative", "writing", "design"]:
            return "creative_flow"
        elif task_type in ["meditation", "relaxation"]:
            return "meditation"
        elif stress_norm > 0.5:
            return "relaxed"
        else:
            # Default balanced state
            return "relaxed"

    @staticmethod
    def get_audio_for_brainwave(brainwave_type):
        """Get audio preset for specific brainwave type"""
        mapping = {
            'delta': 'deep_sleep',
            'theta': 'meditation',
            'alpha': 'relaxed',
            'beta': 'focused',
            'gamma': 'peak_performance'
        }
        return mapping.get(brainwave_type.lower(), 'relaxed')





# ============================================================================
# 6. MAIN FUNCTION FOR TESTING
# ============================================================================

def test_audio_engine():
    """Test the audio engine"""
    print("\n" + "=" * 60)
    print("🎵 NEURO-RHYTHM AI - AUDIO ENGINE TEST")
    print("=" * 60)

    therapist = AudioTherapist()

    try:
        # Test 1: Play a single preset
        print("\n🔊 Test 1: Playing 'focused' preset (16Hz Beta waves)")
        print("   You should hear 16Hz binaural beats")
        print("   Duration: 15 seconds")

        therapist.play_preset("focused", duration_seconds=15)
        time.sleep(17)  # Let it play

        # Test 2: Another preset
        print("\n\n🔊 Test 2: Playing 'meditation' preset (6Hz Theta waves)")
        print("   You should hear 6Hz binaural beats")

        therapist.play_preset("meditation", duration_seconds=15)
        time.sleep(17)

        # Test 3: Isochronic tones
        print("\n\n🔊 Test 3: Playing 'peak_performance' (40Hz Gamma pulses)")
        print("   You should hear pulsed 40Hz tones")

        therapist.play_preset("peak_performance", duration_seconds=15)
        time.sleep(17)

        # Test 4: Volume control
        print("\n\n🔊 Test 4: Testing volume control")
        therapist.adjust_volume(0.3)
        therapist.play_preset("relaxed", duration_seconds=10)
        time.sleep(12)

        therapist.adjust_volume(0.7)

        # Test 5: Fade out
        print("\n\n🔊 Test 5: Testing fade out")
        therapist.play_preset("creative_flow", duration_seconds=20)
        time.sleep(5)  # Play for 5 seconds
        therapist.fade_out(fade_duration=3.0)

        # Test 6: Save to file
        print("\n\n💾 Test 6: Saving audio to file")
        therapist.save_audio_to_file("focused", duration_seconds=5, filename="test_focused.wav")

        print("\n✅ All audio tests completed!")
        print("\n🎧 For best results:")
        print("   1. Use stereo headphones")
        print("   2. Find a quiet environment")
        print("   3. Adjust volume to comfortable level")
        print("   4. Close your eyes and focus on the tones")

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    finally:
        therapist.close()


def quick_demo():
    """Quick demo of all brainwave frequencies"""
    print("\n🎧 BRAINWAVE FREQUENCY DEMO")
    print("   Each frequency will play for 8 seconds")
    print("   Listen for the different 'feeling' of each frequency\n")

    frequencies = [
        ("Delta", 2.5, "deep_sleep"),
        ("Theta", 6.0, "meditation"),
        ("Alpha", 10.0, "relaxed"),
        ("Beta", 16.0, "focused"),
        ("Gamma", 40.0, "peak_performance")
    ]

    therapist = AudioTherapist()

    try:
        for name, freq, preset in frequencies:
            print(f"\n▶️ {name} Waves: {freq} Hz")
            print(f"   Associated with: {AUDIO_PRESETS[preset]['description']}")

            therapist.play_preset(preset, duration_seconds=8)
            time.sleep(10)  # 8 seconds play + 2 second buffer

        print("\n✅ Demo complete!")
        print("   Notice how each frequency feels different:")
        print("   • Delta (2.5Hz): Deep, slow, sleepy")
        print("   • Theta (6Hz): Dreamy, creative, meditative")
        print("   • Alpha (10Hz): Relaxed, calm, focused")
        print("   • Beta (16Hz): Alert, active, thinking")
        print("   • Gamma (40Hz): Intense, focused, high cognition")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted")
    finally:
        therapist.close()


# ============================================================================
# 7. INTEGRATION WITH RL AND VISION
# ============================================================================

class IntegratedAudioSystem:
    """
    Complete audio system ready for integration with RL and vision
    """

    def __init__(self):
        self.therapist = AudioTherapist()
        self.mapper = CognitiveAudioMapper()
        self.current_state = None

    def update_from_vision(self, vision_metrics):
        """
        Update audio based on vision metrics
        vision_metrics should be dict with: focus, stress, drowsiness, task
        """
        if not vision_metrics:
            return

        # Get audio recommendation
        preset_name = self.mapper.map_state_to_audio(
            vision_metrics.get('focus', 50),
            vision_metrics.get('stress', 30),
            vision_metrics.get('drowsiness', 20),
            vision_metrics.get('task', 'unknown')
        )

        # Only change if different from current
        if (self.therapist.current_preset != preset_name and
                preset_name != "silence"):
            print(f"\n🧠 Cognitive State Update:")
            print(f"   Focus: {vision_metrics.get('focus', 0):.0f}%")
            print(f"   Stress: {vision_metrics.get('stress', 0):.0f}%")
            print(f"   Drowsiness: {vision_metrics.get('drowsiness', 0):.0f}%")
            print(f"   Task: {vision_metrics.get('task', 'unknown')}")
            print(f"   → Recommended Audio: {preset_name.replace('_', ' ').title()}")

            self.therapist.play_preset(preset_name, duration_seconds=300)

        self.current_state = vision_metrics

    def update_from_rl(self, rl_recommendation):
        """
        Update audio based on RL agent recommendation
        rl_recommendation should be audio preset name
        """
        if rl_recommendation and rl_recommendation in AUDIO_PRESETS:
            if self.therapist.current_preset != rl_recommendation:
                print(f"\n🤖 RL Agent Recommendation: {rl_recommendation}")
                self.therapist.play_preset(rl_recommendation, duration_seconds=300)

    def handle_voice_command(self, command):
        """
        Handle voice commands like "I need to focus" or "I'm stressed"
        """
        command = command.lower()

        if "focus" in command or "concentrate" in command:
            self.therapist.play_preset("focused", duration_seconds=300)
            return "Switching to focused mode (Beta waves)"

        elif "stress" in command or "anxious" in command or "calm" in command:
            self.therapist.play_preset("stress_relief", duration_seconds=300)
            return "Switching to stress relief (Theta-Alpha blend)"

        elif "creative" in command or "ideas" in command:
            self.therapist.play_preset("creative_flow", duration_seconds=300)
            return "Switching to creative flow (Schumann resonance)"

        elif "sleep" in command or "tired" in command:
            self.therapist.play_preset("deep_sleep", duration_seconds=300)
            return "Switching to deep sleep (Delta waves)"

        elif "stop" in command or "silence" in command:
            self.therapist.stop()
            return "Audio stopped"

        elif "test" in command:
            self.therapist.play_test_sequence(duration_per_preset=5)
            return "Playing test sequence"

        return "Command not recognized"


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neuro-Rhythm Audio Engine")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["test", "demo", "play", "save", "integrate"],
                        help="Operation mode")
    parser.add_argument("--preset", type=str, default="relaxed",
                        help="Preset to play (for play mode)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration in seconds")
    parser.add_argument("--volume", type=float, default=0.7,
                        help="Volume level (0.0 to 1.0)")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🎵 NEURO-RHYTHM AI - BRAINWAVE AUDIO ENGINE")
    print("=" * 60)

    if args.mode == "test":
        test_audio_engine()

    elif args.mode == "demo":
        quick_demo()

    elif args.mode == "play":
        therapist = AudioTherapist()
        try:
            therapist.adjust_volume(args.volume)
            print(f"\n▶️ Playing: {args.preset}")
            print(f"   Duration: {args.duration} seconds")
            print(f"   Volume: {args.volume}")

            therapist.play_preset(args.preset, duration_seconds=args.duration)

            # Wait for playback to complete or user interrupt
            print("\n⏳ Playing... Press Ctrl+C to stop early")
            try:
                while therapist.is_playing:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⚠️ Stopped by user")
                therapist.stop()

        finally:
            therapist.close()

    elif args.mode == "save":
        therapist = AudioTherapist()
        filename = therapist.save_audio_to_file(
            args.preset,
            duration_seconds=args.duration,
            filename=f"{args.preset}_{args.duration}s.wav"
        )
        therapist.close()
        print(f"\n✅ Audio saved to: {filename}")
        print("   You can play this file with any media player")

    elif args.mode == "integrate":
        print("\n🤖 Integrated Audio System Demo")
        system = IntegratedAudioSystem()

        # Simulate vision updates
        test_states = [
            {"focus": 85, "stress": 20, "drowsiness": 10, "task": "coding"},
            {"focus": 30, "stress": 75, "drowsiness": 15, "task": "unknown"},
            {"focus": 60, "stress": 40, "drowsiness": 50, "task": "creative"},
            {"focus": 45, "stress": 90, "drowsiness": 25, "task": "unknown"}
        ]

        for i, state in enumerate(test_states):
            print(f"\n📊 Test {i + 1}: Simulating vision input...")
            system.update_from_vision(state)
            time.sleep(8)  # Listen to each for 8 seconds

        system.therapist.close()

    print("\n" + "=" * 60)
    print("✅ Audio Engine Ready for Integration")
    print("=" * 60)
