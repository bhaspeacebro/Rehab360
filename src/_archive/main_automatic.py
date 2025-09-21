"""
Rehab360 Pro - Ultimate Version with Complete Automation
Comprehensive voice coaching with LLM integration and automatic startup
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import yaml
import os
import sys
import random
import pyttsx3
from gtts import gTTS
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import pygame
import tempfile

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@dataclass
class VoiceMessage:
    text: str
    priority: int = 5
    category: str = 'general'
    interrupt: bool = False
    delay: float = 0.0

class AutoVoiceCoach:
    """Automatic voice coaching system with comprehensive guidance"""

    def __init__(self):
        self.voice_queue = queue.PriorityQueue()
        self.is_speaking = False
        self.session_active = False
        self.exercise_state = "ready"
        self.rep_count = 0
        self.last_speech_time = 0
        self.speech_cooldowns = {'instruction': 1.5, 'correction': 2.0, 'praise': 3.0}

        # Initialize voice engine
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.voice_engine = 'gtts'
        except:
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 180)
            self.voice_engine.setProperty('volume', 1.0)

        # Start voice processing thread
        self.voice_thread = threading.Thread(target=self._voice_processor, daemon=True)
        self.voice_thread.start()

        # Comprehensive voice library
        self.voice_library = self._create_voice_library()

    def _create_voice_library(self) -> Dict:
        """Create comprehensive voice guidance library"""
        return {
            'welcome': [
                "Welcome to Rehab360 Pro! I'm your AI coach. Let's get started!",
                "Hello! I'm here to guide you through your workout. Let's begin!",
                "Ready to exercise? I'll coach you through every movement!"
            ],
            'setup': [
                "Please stand back so I can see your full body clearly.",
                "Move into the camera frame where I can see your entire body.",
                "Position yourself 6-8 feet from the camera for best tracking."
            ],
            'ready': [
                "Perfect! I can see you clearly. Let's start exercising!",
                "Great! You're in position. Ready to begin!",
                "Excellent! I've got a clear view. Let's work out!"
            ],
            'squat': {
                'start': [
                    "Let's begin with squats. Stand with feet shoulder-width apart.",
                    "Starting squat exercise. Position your feet shoulder-width apart, toes slightly out.",
                    "Ready for squats! Remember to keep your chest up and core engaged."
                ],
                'ready_position': [
                    "Good stance. Now slowly lower down by bending your knees.",
                    "Perfect starting position. Begin your descent, keep your back straight.",
                    "Excellent setup. Start lowering, push your hips back like sitting in a chair."
                ],
                'going_down': [
                    "Going down nicely. Keep lowering to 90 degrees.",
                    "Good descent. Control the movement, don't rush.",
                    "That's it, keep going down slowly and controlled."
                ],
                'at_bottom': [
                    "Perfect depth! Now push through your heels to stand up.",
                    "Great squat depth! Drive up through your heels.",
                    "Excellent bottom position! Press up now, squeeze your glutes."
                ],
                'going_up': [
                    "Rising up well. Push through those heels.",
                    "Good ascent. Keep your core tight.",
                    "Coming up nicely. Almost there."
                ],
                'completed': [
                    "Great rep! That's {}.",
                    "Perfect squat! {} done.",
                    "Excellent form! Rep {} complete."
                ],
                'corrections': {
                    'knees_in': "Keep your knees aligned over your toes, don't let them cave in.",
                    'back_round': "Straighten your back. Keep your chest up and proud.",
                    'too_shallow': "Go deeper. Aim for thighs parallel to the ground.",
                    'too_fast': "Slow down. Control both the descent and ascent.",
                    'heels_up': "Keep your heels on the ground throughout the movement.",
                    'leaning_forward': "You're leaning too far forward. Keep your weight balanced."
                },
                'encouragement': [
                    "You're doing great! Keep it up!",
                    "Excellent work! Your form is improving!",
                    "Fantastic effort! Stay focused!",
                    "You're getting stronger with each rep!",
                    "Outstanding progress! Keep going!"
                ]
            },
            'bicep_curl': {
                'start': [
                    "Let's work on bicep curls. Arms at your sides, palms facing forward.",
                    "Starting bicep curls. Keep your elbows locked at your sides.",
                    "Ready for curls! Remember, no swinging, pure bicep work."
                ],
                'ready_position': [
                    "Good arm position. Begin curling up slowly.",
                    "Perfect setup. Start the curl, keep those elbows still.",
                    "Ready to curl. Lift the weight using only your biceps."
                ],
                'going_up': [
                    "Curling up nicely. Keep those elbows pinned.",
                    "Good lift. Squeeze your biceps at the top.",
                    "Rising well. Control the movement."
                ],
                'at_top': [
                    "Perfect contraction! Now lower slowly.",
                    "Great squeeze! Control the descent.",
                    "Excellent peak! Lower with control."
                ],
                'going_down': [
                    "Lowering well. Resist gravity, don't just drop.",
                    "Good control on the way down. Feel the stretch.",
                    "Nice eccentric phase. Almost at the bottom."
                ],
                'completed': [
                    "Strong curl! That's {}.",
                    "Perfect bicep curl! {} done.",
                    "Great form! Rep {} complete."
                ],
                'corrections': {
                    'swinging': "Stop swinging! Keep your body still, isolate the biceps.",
                    'elbows_forward': "Keep your elbows back at your sides.",
                    'too_fast': "Slow down the movement. Two seconds up, two seconds down.",
                    'partial_rep': "Full range of motion! Extend completely at the bottom.",
                    'using_shoulders': "Don't use your shoulders. This is all biceps."
                },
                'encouragement': [
                    "Your biceps are working hard!",
                    "Great isolation! Feel that burn!",
                    "Excellent control throughout!",
                    "Your arms are getting stronger!",
                    "Perfect tempo! Keep it up!"
                ]
            },
            'shoulder_press': {
                'start': [
                    "Time for shoulder press. Hands at shoulder height, elbows bent.",
                    "Starting shoulder press. Keep your core engaged throughout.",
                    "Ready to press! Remember to press straight up, not forward."
                ],
                'ready_position': [
                    "Good starting position. Press up powerfully.",
                    "Perfect setup. Drive those arms straight up.",
                    "Ready to press. Push through your palms."
                ],
                'going_up': [
                    "Pressing up well. Keep it straight overhead.",
                    "Good drive upward. Don't arch your back.",
                    "Rising nicely. Full extension coming."
                ],
                'at_top': [
                    "Full extension! Now lower with control.",
                    "Perfect lockout! Bring it down slowly.",
                    "Great press! Control the descent."
                ],
                'going_down': [
                    "Lowering well. Keep it controlled.",
                    "Good descent. Back to shoulder level.",
                    "Coming down nicely. Feel those shoulders work."
                ],
                'completed': [
                    "Strong press! That's {}.",
                    "Excellent shoulder press! {} done.",
                    "Perfect form! Rep {} complete."
                ],
                'corrections': {
                    'arching_back': "Don't arch your back! Keep your core tight.",
                    'pressing_forward': "Press straight up, not forward.",
                    'uneven_arms': "Keep both arms moving together evenly.",
                    'partial_lockout': "Full extension at the top! Lock out those arms.",
                    'elbows_flared': "Don't flare your elbows too wide."
                },
                'encouragement': [
                    "Strong shoulders! Keep pressing!",
                    "Excellent power! You've got this!",
                    "Great stability throughout!",
                    "Your shoulders are getting stronger!",
                    "Perfect pressing form!"
                ]
            },
            'transitions': {
                'squat_to_bicep': "Great work on squats! Now let's switch to bicep curls.",
                'bicep_to_shoulder': "Excellent arm work! Now let's do shoulder press.",
                'shoulder_to_squat': "Strong shoulders! Back to squats for lower body.",
                'any_to_squat': "Switching to squats. Get your feet shoulder-width apart.",
                'any_to_bicep': "Now for bicep curls. Arms at your sides.",
                'any_to_shoulder': "Time for shoulder press. Bring hands to shoulder height."
            },
            'milestones': {
                5: "Fantastic! 5 reps done. You're warming up nicely!",
                10: "Incredible! 10 reps completed. You're on fire!",
                15: "Amazing! 15 reps. Your strength is showing!",
                20: "Phenomenal! 20 reps. You're a machine!",
                25: "Outstanding! 25 reps. Your endurance is impressive!"
            }
        }

    def start_session(self):
        """Start the coaching session"""
        self.session_active = True
        welcome = random.choice(self.voice_library['welcome'])
        self.speak(welcome, priority=10, category='instruction', interrupt=True)
        time.sleep(1.5)
        self.speak("Please stand back so I can see your full body.", priority=9, category='setup')

    def guide_exercise(self, exercise: str, state: str, rep_count: int, form_score: float):
        """Provide continuous exercise guidance"""
        if not self.session_active:
            self.start_session()

        current_time = time.time()

        # Handle state changes
        if state != self.exercise_state:
            self.exercise_state = state
            self._handle_state_change(exercise, state, rep_count)

        # Handle rep milestones
        if rep_count in self.voice_library['milestones']:
            self.speak(self.voice_library['milestones'][rep_count], priority=8, category='praise')

        # Handle form corrections
        if form_score < 0.7 and current_time - self.last_speech_time > 3.0:
            self._provide_form_correction(exercise, form_score)
            self.last_speech_time = current_time

        # Handle encouragement
        elif form_score > 0.8 and rep_count > 0 and rep_count % 3 == 0:
            if current_time - self.last_speech_time > 4.0:
                exercise_lib = self.voice_library.get(exercise, {})
                encouragement = random.choice(exercise_lib.get('encouragement', []))
                self.speak(encouragement, priority=5, category='praise')

    def _handle_state_change(self, exercise: str, state: str, rep_count: int):
        """Handle changes in exercise state"""
        if state == "ready" and rep_count == 0:
            # Starting new exercise
            exercise_lib = self.voice_library.get(exercise, {})
            start_msg = random.choice(exercise_lib.get('start', []))
            self.speak(start_msg, priority=9, category='instruction')
        elif state == "ready_position":
            exercise_lib = self.voice_library.get(exercise, {})
            ready_msg = random.choice(exercise_lib.get('ready_position', []))
            self.speak(ready_msg, priority=7, category='instruction')
        elif state == "going_down":
            exercise_lib = self.voice_library.get(exercise, {})
            down_msg = random.choice(exercise_lib.get('going_down', []))
            self.speak(down_msg, priority=6, category='instruction')
        elif state == "at_bottom":
            exercise_lib = self.voice_library.get(exercise, {})
            bottom_msg = random.choice(exercise_lib.get('at_bottom', []))
            self.speak(bottom_msg, priority=7, category='instruction')
        elif state == "going_up":
            exercise_lib = self.voice_library.get(exercise, {})
            up_msg = random.choice(exercise_lib.get('up_msg', []))
            self.speak(up_msg, priority=6, category='instruction')

    def _provide_form_correction(self, exercise: str, form_score: float):
        """Provide form correction based on score"""
        exercise_lib = self.voice_library.get(exercise, {})
        corrections = exercise_lib.get('corrections', {})

        # Select appropriate correction based on form score
        if form_score < 0.5:
            correction = random.choice(list(corrections.values()))
            self.speak(correction, priority=8, category='correction')

    def speak(self, text: str, priority: int = 5, category: str = 'general',
             interrupt: bool = False, delay: float = 0.0):
        """Queue a voice message"""
        message = VoiceMessage(text, priority, category, interrupt, delay)
        self.voice_queue.put((10 - priority, time.time(), message))  # Lower priority number = higher priority

        if interrupt:
            self._clear_queue()

    def _clear_queue(self):
        """Clear all pending voice messages"""
        while not self.voice_queue.empty():
            try:
                self.voice_queue.get_nowait()
            except queue.Empty:
                break

    def _voice_processor(self):
        """Background thread for processing voice messages"""
        while True:
            try:
                _, timestamp, message = self.voice_queue.get(timeout=0.1)

                if message.delay > 0:
                    time.sleep(message.delay)

                self._speak_text(message.text)
                self.last_speech_time = time.time()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice error: {e}")
                time.sleep(0.5)

    def _speak_text(self, text: str):
        """Actually speak the text"""
        self.is_speaking = True

        try:
            if self.voice_engine == 'gtts':
                tts = gTTS(text=text, lang='en', slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tts.save(tmp_file.name)
                    pygame.mixer.music.load(tmp_file.name)
                    pygame.mixer.music.play()

                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)

                    pygame.mixer.music.unload()
                    os.unlink(tmp_file.name)
            else:
                self.voice_engine.say(text)
                self.voice_engine.runAndWait()

        except Exception as e:
            print(f"Speech error: {e}")
            print(f"[Coach]: {text}")
        finally:
            self.is_speaking = False

    def end_session(self):
        """End the coaching session"""
        self.session_active = False
        self._clear_queue()

    def shutdown(self):
        """Clean shutdown"""
        self.session_active = False
        self._clear_queue()

        try:
            if self.voice_engine == 'gtts':
                pygame.mixer.quit()
            elif hasattr(self.voice_engine, 'stop'):
                self.voice_engine.stop()
        except:
            pass

class PoseSkeleton:
    """Enhanced pose skeleton drawing"""

    def __init__(self):
        self.connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]

    def draw_skeleton(self, frame: np.ndarray, landmarks: List[Tuple[float, float]]) -> np.ndarray:
        """Draw professional skeleton overlay"""
        if not landmarks or len(landmarks) < 33:
            return frame

        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame, dtype=np.uint8)

        # Convert normalized to pixel coordinates
        landmarks_px = []
        for i, (x, y) in enumerate(landmarks):
            px_x = int(x * w)
            px_y = int(y * h)
            landmarks_px.append((px_x, px_y))

        # Draw bones
        for start_idx, end_idx in self.connections:
            if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                start_pt = landmarks_px[start_idx]
                end_pt = landmarks_px[end_idx]

                cv2.line(overlay, start_pt, end_pt, (255, 255, 255), 3, cv2.LINE_AA)

                # Add glow effect
                glow_offset = 2
                shadow_start = (start_pt[0] + glow_offset, start_pt[1] + glow_offset)
                shadow_end = (end_pt[0] + glow_offset, end_pt[1] + glow_offset)
                cv2.line(overlay, shadow_start, shadow_end, (50, 50, 50), 4, cv2.LINE_AA)

        # Draw joints
        joint_indices = set()
        for start_idx, end_idx in self.connections:
            joint_indices.add(start_idx)
            joint_indices.add(end_idx)

        for idx in joint_indices:
            if idx < len(landmarks_px):
                center = landmarks_px[idx]

                # Outer glow
                cv2.circle(overlay, center, 10, (100, 200, 255), -1, cv2.LINE_AA)
                # White border
                cv2.circle(overlay, center, 8, (255, 255, 255), 2, cv2.LINE_AA)
                # Inner joint
                cv2.circle(overlay, center, 6, (255, 100, 100), -1, cv2.LINE_AA)
                # Center highlight
                cv2.circle(overlay, center, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Blend with frame
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        return result

class MotionAnalyzer:
    """Real-time motion analysis"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True
        )

        self.last_landmarks = None
        self.angle_cache = {}
        self.last_angle_time = 0

    def process_frame(self, frame: np.ndarray) -> Optional[List]:
        """Process frame and return landmarks"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z, lm.visibility)
                        for lm in results.pose_landmarks.landmark]
            self.last_landmarks = landmarks
            return landmarks

        return self.last_landmarks

    def calculate_angles(self, landmarks: List, exercise: str) -> Dict[str, float]:
        """Calculate exercise-specific angles"""
        if not landmarks or len(landmarks) < 33:
            return {}

        current_time = time.time()
        if current_time - self.last_angle_time < 0.05:
            return self.angle_cache

        angles = {}

        try:
            if exercise == "squat":
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
                r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                angles['hip'] = self._calculate_angle(r_shoulder, r_hip, r_knee)
                angles['knee'] = self._calculate_angle(r_hip, r_knee, r_ankle)

            elif exercise == "bicep curl":
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

                angles['elbow'] = self._calculate_angle(r_shoulder, r_elbow, r_wrist)

            elif exercise == "shoulder press":
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

                angles['elbow'] = self._calculate_angle(r_shoulder, r_elbow, r_wrist)

        except (IndexError, TypeError):
            return self.angle_cache

        self.angle_cache = angles
        self.last_angle_time = current_time
        return angles

    def _calculate_angle(self, a, b, c) -> float:
        """Calculate angle between three points"""
        if not all([a, b, c]):
            return 0

        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)

    def cleanup(self):
        """Clean up resources"""
        self.pose.close()

class ExerciseDetector:
    """Detect exercise phases and form"""

    def __init__(self):
        self.phase_thresholds = {
            'squat': {'down_start': 150, 'bottom': 100, 'up_start': 110, 'top': 160},
            'bicep curl': {'up_start': 140, 'top': 50, 'down_start': 60, 'bottom': 150},
            'shoulder press': {'up_start': 100, 'top': 160, 'down_start': 150, 'bottom': 90}
        }

    def detect_phase(self, exercise: str, angles: Dict[str, float], previous_phase: str) -> str:
        """Detect current exercise phase"""
        if not angles:
            return "unknown"

        primary_angle = None
        if exercise == "squat" and 'knee' in angles:
            primary_angle = angles['knee']
        elif exercise in ["bicep curl", "shoulder press"] and 'elbow' in angles:
            primary_angle = angles['elbow']

        if primary_angle is None:
            return previous_phase

        thresholds = self.phase_thresholds.get(exercise, {})

        if exercise == "squat":
            if primary_angle > thresholds.get('top', 160):
                return "ready"
            elif primary_angle < thresholds.get('bottom', 100):
                return "bottom"
            elif previous_phase == "ready" and primary_angle < thresholds.get('down_start', 150):
                return "down"
            elif previous_phase == "bottom" and primary_angle > thresholds.get('up_start', 110):
                return "up"

        elif exercise == "bicep curl":
            if primary_angle > thresholds.get('bottom', 150):
                return "ready"
            elif primary_angle < thresholds.get('top', 50):
                return "top"
            elif previous_phase == "ready" and primary_angle < thresholds.get('up_start', 140):
                return "up"
            elif previous_phase == "top" and primary_angle > thresholds.get('down_start', 60):
                return "down"

        elif exercise == "shoulder press":
            if primary_angle < thresholds.get('bottom', 90):
                return "ready"
            elif primary_angle > thresholds.get('top', 160):
                return "top"
            elif previous_phase == "ready" and primary_angle > thresholds.get('up_start', 100):
                return "up"
            elif previous_phase == "top" and primary_angle < thresholds.get('down_start', 150):
                return "down"

        return previous_phase

    def calculate_form_score(self, exercise: str, angles: Dict[str, float]) -> float:
        """Calculate form score based on angles"""
        if not angles:
            return 0.5

        ideal_angles = {
            'squat': {'knee': 90, 'hip': 90},
            'bicep curl': {'elbow': 45},
            'shoulder press': {'elbow': 90}
        }

        if exercise not in ideal_angles:
            return 0.5

        scores = []
        for joint, ideal in ideal_angles[exercise].items():
            if joint in angles:
                deviation = abs(angles[joint] - ideal)
                score = max(0, 1 - deviation / 90)
                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.5

    def detect_rep_completion(self, exercise: str, current_phase: str, previous_phase: str) -> bool:
        """Detect if a rep was completed"""
        if exercise == "squat":
            return previous_phase in ["bottom", "up"] and current_phase == "ready"
        elif exercise == "bicep curl":
            return previous_phase in ["top", "down"] and current_phase == "ready"
        elif exercise == "shoulder press":
            return previous_phase in ["top", "down"] and current_phase == "ready"
        return False

class Rehab360Pro:
    """Main application class"""

    def __init__(self):
        print("=" * 70)
        print("🏥 REHAB360 PRO - Automatic Motion Analysis System")
        print("   With Complete Voice Coaching Integration")
        print("=" * 70)

        # Initialize components
        self.motion_analyzer = MotionAnalyzer()
        self.exercise_detector = ExerciseDetector()
        self.pose_skeleton = PoseSkeleton()
        self.voice_coach = AutoVoiceCoach()

        # Exercise state
        self.current_exercise = "squat"
        self.rep_count = 0
        self.current_phase = "unknown"
        self.previous_phase = "unknown"
        self.form_score = 1.0

        # UI state
        self.running = True
        self.show_skeleton = True
        self.voice_enabled = True

        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()

        print("✅ System initialized successfully!")
        print("-" * 70)

    def setup_camera(self) -> cv2.VideoCapture:
        """Setup camera with optimal settings"""
        print("📷 Setting up camera...")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("❌ ERROR: Camera not found!")
            return None

        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("✅ Camera ready!")
        return cap

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame"""
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Get pose landmarks
        landmarks = self.motion_analyzer.process_frame(frame)

        if landmarks:
            # Person detected - start coaching if not already started
            if not self.voice_coach.session_active:
                self.voice_coach.start_session()
                self.voice_coach.speak("Perfect! I can see you clearly. Let's start exercising!", priority=9)

            # Convert to 2D for drawing
            landmarks_2d = [(lm[0], lm[1]) for lm in landmarks]

            # Calculate angles
            angles = self.motion_analyzer.calculate_angles(landmarks, self.current_exercise)

            # Detect phase
            self.previous_phase = self.current_phase
            self.current_phase = self.exercise_detector.detect_phase(
                self.current_exercise, angles, self.previous_phase
            )

            # Calculate form score
            self.form_score = self.exercise_detector.calculate_form_score(
                self.current_exercise, angles
            )

            # Check for rep completion
            if self.exercise_detector.detect_rep_completion(
                self.current_exercise, self.current_phase, self.previous_phase
            ):
                self.rep_count += 1

                # Provide rep feedback
                if self.voice_enabled:
                    self.voice_coach.speak(f"Great rep! That's {self.rep_count}.", priority=7)

            # Provide continuous guidance
            if self.voice_enabled:
                self.voice_coach.guide_exercise(
                    self.current_exercise, self.current_phase, self.rep_count, self.form_score
                )

            # Draw skeleton
            if self.show_skeleton:
                frame = self.pose_skeleton.draw_skeleton(frame, landmarks_2d)

        else:
            # No person detected
            if self.voice_coach.session_active:
                self.voice_coach.speak("I don't see you. Please step into the camera view.", priority=6)

        # Draw UI
        frame = self._draw_ui(frame)

        # Update FPS
        self._update_fps()

        return frame

    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay"""
        h, w = frame.shape[:2]

        # Top panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Title
        cv2.putText(frame, "REHAB360 PRO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # Exercise info
        cv2.putText(frame, f"Exercise: {self.current_exercise.upper()}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Reps: {self.rep_count}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Form score
        form_color = (0, 255, 0) if self.form_score > 0.7 else (0, 255, 255)
        if self.form_score < 0.5:
            form_color = (0, 0, 255)
        cv2.putText(frame, f"Form: {self.form_score".0%"}", (200, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)

        # Phase
        cv2.putText(frame, f"Phase: {self.current_phase.upper()}", (350, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # FPS
        avg_fps = self._get_avg_fps()
        cv2.putText(frame, f"FPS: {avg_fps".1f"}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Voice status
        voice_status = "Voice: ON" if self.voice_enabled else "Voice: OFF"
        voice_color = (0, 255, 0) if self.voice_enabled else (128, 128, 128)
        cv2.putText(frame, voice_status, (w - 150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, voice_color, 2)

        # Controls
        controls = "S:Squat | B:Bicep | P:Press | V:Voice | K:Skeleton | R:Reset | Q:Quit"
        cv2.putText(frame, controls, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.fps_history.append(frame_time)
        self.last_frame_time = current_time

    def _get_avg_fps(self) -> float:
        """Get average FPS"""
        if not self.fps_history:
            return 0.0
        avg_time = sum(self.fps_history) / len(self.fps_history)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def handle_keyboard(self, key: int):
        """Handle keyboard input"""
        if key == ord('q'):
            self.running = False
        elif key == ord('s'):
            self.switch_exercise("squat")
        elif key == ord('b'):
            self.switch_exercise("bicep curl")
        elif key == ord('p'):
            self.switch_exercise("shoulder press")
        elif key == ord('v'):
            self.voice_enabled = not self.voice_enabled
            print(f"Voice: {'ON' if self.voice_enabled else 'OFF'}")
        elif key == ord('k'):
            self.show_skeleton = not self.show_skeleton
            print(f"Skeleton: {'ON' if self.show_skeleton else 'OFF'}")
        elif key == ord('r'):
            self.rep_count = 0
            print("Counter reset!")

    def switch_exercise(self, exercise: str):
        """Switch exercise with voice guidance"""
        if exercise != self.current_exercise:
            self.current_exercise = exercise
            self.rep_count = 0
            self.current_phase = "unknown"
            self.previous_phase = "unknown"

            if self.voice_enabled:
                self.voice_coach.speak(f"Switching to {exercise.replace('_', ' ')}", priority=9, interrupt=True)

    def run(self):
        """Main application loop"""
        cap = self.setup_camera()
        if not cap:
            return

        print("\n🎯 Starting Rehab360 Pro...")
        print("System will automatically:")
        print("  • Guide you through exercises")
        print("  • Provide real-time feedback")
        print("  • Track your form and reps")
        print("  • Give voice corrections")
        print("\nPress 'H' for help or 'Q' to quit\n")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera read failed!")
                    break

                # Process frame
                frame = self.process_frame(frame)

                # Display
                cv2.imshow("Rehab360 Pro - Automatic Motion Analysis", frame)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self.handle_keyboard(key)

        except KeyboardInterrupt:
            print("\n⏹ Session ended by user")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup(cap)

    def cleanup(self, cap: cv2.VideoCapture):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")

        # End voice session
        if self.voice_coach:
            self.voice_coach.end_session()
            time.sleep(2)  # Let final message play
            self.voice_coach.shutdown()

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.motion_analyzer.cleanup()

        print("✅ Thank you for using Rehab360 Pro!")

def main():
    """Main entry point"""
    app = Rehab360Pro()
    app.run()

if __name__ == "__main__":
    main()
