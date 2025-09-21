"""
Rehab360 Pro - Complete Motion Analysis System
Enhanced with real-time voice coaching, professional skeleton visualization, and comprehensive exercise analysis
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import sys
import os
import threading
import queue
import random
import tempfile
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from collections import deque

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Optional TTS imports
try:
    from gtts import gTTS
    import pygame
    _HAS_GTTs = True
    _HAS_PYGAME = True
except ImportError:
    _HAS_GTTs = False
    _HAS_PYGAME = False

try:
    import pyttsx3
    _HAS_PYTT = True
except ImportError:
    _HAS_PYTT = False

# Import pose detection and visualization
try:
    from pose_draw import PoseSkeleton, AngleVisualizer, ExerciseAnalyzer
except ImportError:
    print("Warning: pose_draw module not found. Using basic functionality.")

@dataclass
class VoiceMessage:
    """Structured voice message with priority"""
    text: str
    priority: int = 5
    category: str = "general"
    interrupt: bool = False
    delay: float = 0.0

class EnhancedVoiceCoach:
    """
    Reliable background voice coach with comprehensive exercise guidance
    """

    def __init__(self, prefer_gtts: bool = True):
        # Select engine
        self.voice_engine_type = "none"
        self.engine: Optional[Any] = None

        if prefer_gtts and _HAS_GTTs and _HAS_PYGAME:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.voice_engine_type = "gtts"
            except Exception:
                self.voice_engine_type = "none"

        if self.voice_engine_type == "none" and _HAS_PYTT:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 170)
                self.engine.setProperty("volume", 0.9)
                # Try set female voice if available
                try:
                    voices = self.engine.getProperty("voices")
                    for v in voices:
                        if "female" in getattr(v, "name", "").lower():
                            self.engine.setProperty("voice", v.id)
                            break
                except Exception:
                    pass
                self.voice_engine_type = "pyttsx3"
            except Exception:
                self.voice_engine_type = "none"

        if self.voice_engine_type == "none":
            print("Warning: No TTS engine available. Voice coach will print messages.")

        # Queue and thread
        self._q: "queue.PriorityQueue" = queue.PriorityQueue()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        # State
        self.session_active = False
        self.last_speech_time = 0.0
        self.last_rep = 0
        self.exercise_state = {"position": "not_visible", "rep_phase": "neutral", "last_feedback": 0.0}

        # Build comprehensive voice library
        self.voice_library = self._load_voice_library()

    def _load_voice_library(self) -> Dict:
        """Load comprehensive voice guidance library"""
        return {
            "general": {
                "welcome": [
                    "Welcome to Rehab360 Pro! I'm your personal coach.",
                    "Hello! Let's begin your exercise session.",
                    "Ready to start? I'll guide you through each movement."
                ],
                "no_person": [
                    "I don't see you. Please step into the camera view.",
                    "Stand in front of the camera so I can see you.",
                    "Position yourself where I can track your movements."
                ],
                "person_detected": [
                    "Perfect, I can see you. Let's begin your workout!",
                    "Great! I have you in view. Ready to exercise?",
                    "Excellent! You're in position. Let's start!"
                ],
                "session_end": [
                    "Great session! You completed {} reps total.",
                    "Excellent workout! You did {} reps today.",
                    "Fantastic job! {} reps completed in this session."
                ]
            },
            "squat": {
                "start": [
                    "Starting squats. Feet shoulder-width apart, toes slightly out.",
                    "Let's begin with squats. Stand with feet hip-width apart.",
                    "Ready for squats! Keep your chest up and core engaged."
                ],
                "positioning": [
                    "Stand with feet shoulder-width apart.",
                    "Keep your toes slightly pointed outward.",
                    "Maintain an upright posture with chest lifted."
                ],
                "ready_position": [
                    "Good stance. Now slowly lower down by bending your knees.",
                    "Perfect setup. Begin your descent, push your hips back.",
                    "Excellent position. Start lowering, like sitting in a chair."
                ],
                "going_down": [
                    "Going down nicely. Keep lowering to 90 degrees.",
                    "Good descent. Control the movement, don't rush.",
                    "That's it, keep going down slowly and controlled."
                ],
                "at_bottom": [
                    "Perfect depth! Now push through your heels to stand up.",
                    "Great squat depth! Drive up through your heels.",
                    "Excellent bottom position! Press up now, squeeze your glutes."
                ],
                "going_up": [
                    "Rising up well. Push through those heels.",
                    "Good ascent. Keep your core tight.",
                    "Coming up nicely. Almost there."
                ],
                "completed_rep": [
                    "Great rep! That's {}.",
                    "Perfect squat! {} done.",
                    "Excellent form! Rep {} complete."
                ],
                "form_corrections": {
                    'knees_in': "Keep your knees aligned over your toes, don't let them cave in.",
                    'back_round': "Straighten your back. Keep your chest up and proud.",
                    'too_shallow': "Go deeper. Aim for thighs parallel to the ground.",
                    'too_fast': "Slow down. Control both the descent and ascent.",
                    'heels_up': "Keep your heels on the ground throughout the movement.",
                    'leaning_forward': "You're leaning too far forward. Keep your weight balanced."
                },
                'depth_feedback': {
                    'too_shallow': "Go lower! Aim for thighs parallel to the ground.",
                    'perfect_depth': "Perfect depth! Thighs are parallel to the ground.",
                    'too_deep': "Good depth, but don't go too low to avoid strain."
                },
                'encouragement': [
                    "You're doing great! Keep it up!",
                    "Excellent work! Your form is improving!",
                    "Fantastic effort! Stay focused!",
                    "You're getting stronger with each rep!",
                    "Outstanding progress! Keep going!",
                    "Perfect technique! You're a natural!",
                    "Great control throughout the movement!"
                ],
                'milestone': {
                    5: "Fantastic! 5 squats done. You're warming up nicely!",
                    10: "Incredible! 10 squats completed. You're on fire!",
                    15: "Amazing! 15 squats. Your legs are getting a great workout!",
                    20: "Phenomenal! 20 squats. You're a squat machine!",
                    25: "Outstanding! 25 squats. Your endurance is impressive!",
                    30: "Unbelievable! 30 squats. You're crushing it!",
                    50: "Legendary! 50 squats completed. You're unstoppable!"
                }
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
                'completed_rep': [
                    "Strong curl! That's {}.",
                    "Perfect bicep curl! {} done.",
                    "Great form! Rep {} complete."
                ],
                'form_corrections': {
                    'swinging': "Stop swinging! Keep your body still, isolate the biceps.",
                    'elbows_forward': "Keep your elbows back at your sides.",
                    'too_fast': "Slow down the movement. Two seconds up, two seconds down.",
                    'partial_rep': "Full range of motion! Extend completely at the bottom.",
                    'using_shoulders': "Don't use your shoulders. This is all biceps.",
                    'leaning_back': "Don't lean back. Keep your torso stable."
                },
                'encouragement': [
                    "Your biceps are working hard!",
                    "Excellent control on the negative!",
                    "Great mind-muscle connection!",
                    "Perfect tempo throughout!",
                    "You're building serious strength!",
                    "Outstanding bicep isolation!",
                    "Beautiful curl execution!"
                ],
                'milestone': {
                    5: "Great start! 5 bicep curls completed.",
                    10: "Excellent! 10 curls done. Keep that control!",
                    15: "Impressive! 15 curls. Your biceps are firing!",
                    20: "Outstanding! 20 curls. You're building real strength!",
                    25: "Amazing! 25 curls. Your arms are getting pumped!",
                    30: "Phenomenal! 30 curls. You're a bicep beast!",
                    50: "Legendary! 50 curls completed. Unbelievable!"
                }
            },
            'shoulder_press': {
                'start': [
                    "Let's do shoulder presses. Weights at shoulder height.",
                    "Starting shoulder press. Keep your core tight.",
                    "Ready for presses! Lock in that core stability."
                ],
                'ready_position': [
                    "Good setup. Begin pressing overhead.",
                    "Perfect position. Press the weights up smoothly.",
                    "Ready to press. Drive through your shoulders."
                ],
                'going_up': [
                    "Pressing up well. Keep those wrists straight.",
                    "Good drive. Lock out at the top.",
                    "Rising smoothly. Control the movement."
                ],
                'at_top': [
                    "Perfect lockout! Now lower with control.",
                    "Great extension! Lower slowly.",
                    "Excellent overhead position! Descend now."
                ],
                'going_down': [
                    "Lowering well. Keep tension in your shoulders.",
                    "Good control. Almost back to start.",
                    "Nice descent. Prepare for next rep."
                ],
                'completed_rep': [
                    "Strong press! That's {}.",
                    "Perfect shoulder press! {} done.",
                    "Excellent form! Rep {} complete."
                ],
                'form_corrections': {
                    'arching_back': "Don't arch your back. Keep your core braced.",
                    'leaning_forward': "Don't lean forward. Stay upright.",
                    'incomplete_lockout': "Lock out completely at the top.",
                    'too_fast': "Slow down. Control the entire movement.",
                    'elbows_flared': "Keep elbows at 45 degrees, not flared out.",
                    'dropping_elbows': "Don't drop your elbows too low."
                },
                'encouragement': [
                    "Great shoulder stability!",
                    "Excellent overhead strength!",
                    "Perfect pressing mechanics!",
                    "Outstanding core control!",
                    "You're building powerful shoulders!",
                    "Beautiful press execution!",
                    "Impressive overhead control!"
                ],
                'milestone': {
                    5: "Great start! 5 shoulder presses completed.",
                    10: "Excellent! 10 presses done. Keep that stability!",
                    15: "Impressive! 15 presses. Your shoulders are working!",
                    20: "Outstanding! 20 presses. You're building real power!",
                    25: "Amazing! 25 presses. Your shoulders are getting strong!",
                    30: "Phenomenal! 30 presses. You're a pressing powerhouse!",
                    50: "Legendary! 50 presses completed. Extraordinary!"
                }
            }
        }

    def speak(self, text: str, priority: int = 5, category: str = "general", interrupt: bool = False, delay: float = 0.0):
        """Queue a message. Lower priority tuple sorts earlier in queue."""
        if not text:
            return
        msg = VoiceMessage(text=text, priority=priority, category=category, interrupt=interrupt, delay=delay)
        queue_priority = (10 - int(priority), time.time())
        if interrupt:
            self.clear_queue()
            self._stop_current_playback()
        self._q.put((queue_priority, msg))

    def guide_exercise(self, exercise: str, angles: Dict[str, float], rep_count: int, state: str, form_score: float):
        """Provide comprehensive real-time exercise guidance"""
        try:
            if not self.session_active:
                self.session_active = True
                welcome = random.choice(self.voice_library["general"]["welcome"])
                self.speak(welcome, priority=9, category="general", interrupt=True)

            # Exercise-specific guidance
            if exercise.lower() in self.voice_library:
                ex_data = self.voice_library[exercise.lower()]

                # Rep completion with milestone celebrations
                if rep_count > self.last_rep:
                    self.last_rep = rep_count

                    # Milestone messages
                    if rep_count in ex_data.get('milestone', {}):
                        milestone_msg = ex_data['milestone'][rep_count]
                        self.speak(milestone_msg, priority=8, category="milestone")
                    else:
                        # Regular rep completion
                        rep_msgs = ex_data.get('completed_rep', ["Great rep! That's {}."])
                        rep_msg = random.choice(rep_msgs).format(rep_count)
                        self.speak(rep_msg, priority=7, category="rep_completion")

                # Form corrections based on state and angles
                if form_score < 0.7:
                    corrections = ex_data.get('form_corrections', {})
                    if state in corrections:
                        self.speak(corrections[state], priority=8, category="correction")

                # Exercise-specific feedback
                if exercise.lower() == 'squat':
                    self._provide_squat_feedback(angles, ex_data)
                elif exercise.lower() == 'bicep_curl':
                    self._provide_bicep_feedback(angles, ex_data)
                elif exercise.lower() == 'shoulder_press':
                    self._provide_press_feedback(angles, ex_data)

                # Random encouragement (10% chance per call)
                if random.random() < 0.1:
                    encouragement = ex_data.get('encouragement', ["Good work!"])
                    self.speak(random.choice(encouragement), priority=5, category="encouragement")

        except Exception as e:
            print(f"Error in guide_exercise: {e}")

    def _provide_squat_feedback(self, angles: Dict[str, float], ex_data: Dict):
        """Provide detailed squat-specific feedback"""
        try:
            knee_angles = []
            if 'left_knee' in angles:
                knee_angles.append(angles['left_knee'])
            if 'right_knee' in angles:
                knee_angles.append(angles['right_knee'])

            if knee_angles:
                avg_knee_angle = sum(knee_angles) / len(knee_angles)

                # Depth feedback
                if 85 <= avg_knee_angle <= 95:
                    depth_msg = ex_data.get('depth_feedback', {}).get('perfect_depth', "Perfect squat depth!")
                    self.speak(depth_msg, priority=6, category="form_feedback")
                elif avg_knee_angle > 100:
                    depth_msg = ex_data.get('depth_feedback', {}).get('too_shallow', "Go lower for better results!")
                    self.speak(depth_msg, priority=7, category="correction")
                elif avg_knee_angle < 80:
                    depth_msg = ex_data.get('depth_feedback', {}).get('too_deep', "Good depth, but don't go too low!")
                    self.speak(depth_msg, priority=6, category="form_feedback")

        except Exception as e:
            print(f"Error in squat feedback: {e}")

    def _provide_bicep_feedback(self, angles: Dict[str, float], ex_data: Dict):
        """Provide detailed bicep curl feedback"""
        try:
            elbow_angles = []
            if 'left_elbow' in angles:
                elbow_angles.append(angles['left_elbow'])
            if 'right_elbow' in angles:
                elbow_angles.append(angles['right_elbow'])

            if elbow_angles:
                avg_elbow_angle = sum(elbow_angles) / len(elbow_angles)

                # Range feedback
                if avg_elbow_angle < 45:
                    self.speak("Full contraction! Squeeze those biceps hard!", priority=6, category="form_feedback")
                elif avg_elbow_angle > 150:
                    self.speak("Lower the weight completely for full range!", priority=7, category="correction")

        except Exception as e:
            print(f"Error in bicep feedback: {e}")

    def _provide_press_feedback(self, angles: Dict[str, float], ex_data: Dict):
        """Provide detailed shoulder press feedback"""
        try:
            elbow_angles = []
            if 'left_elbow' in angles:
                elbow_angles.append(angles['left_elbow'])
            if 'right_elbow' in angles:
                elbow_angles.append(angles['right_elbow'])

            if elbow_angles:
                avg_elbow_angle = sum(elbow_angles) / len(elbow_angles)

                # Lockout feedback
                if avg_elbow_angle < 20:
                    self.speak("Perfect lockout! Full extension achieved!", priority=6, category="form_feedback")
                elif avg_elbow_angle > 30:
                    self.speak("Lock out completely at the top!", priority=7, category="correction")

        except Exception as e:
            print(f"Error in press feedback: {e}")

    def handle_exercise_transition(self, new_exercise: str):
        """Handle smooth transition to a new exercise with voice guidance"""
        try:
            exercise_name = new_exercise.lower()
            if exercise_name in self.voice_library:
                start_msgs = self.voice_library[exercise_name].get('start', [])
                if start_msgs:
                    transition_msg = f"Now switching to {new_exercise}. " + random.choice(start_msgs)
                    self.speak(transition_msg, priority=9, category="transition", interrupt=True)
                else:
                    self.speak(f"Switching to {new_exercise}.", priority=8, category="transition")
            else:
                self.speak(f"Starting {new_exercise}.", priority=8, category="transition")
        except Exception as e:
            print(f"Error in exercise transition: {e}")

    def handle_no_person_detected(self):
        """Queue reminder periodically when no person is present"""
        now = time.time()
        if now - self.last_speech_time > 5.0:
            self.speak(random.choice(self.voice_library["general"]["no_person"]), priority=6, category="general")
            self.last_speech_time = now

    def handle_person_detected(self):
        """Announce when person appears"""
        self.speak(random.choice(self.voice_library["general"]["person_detected"]), priority=7, category="general")

    def end_session(self, total_reps: int):
        """Announce session end and clear queue"""
        self.clear_queue()
        end_msg = random.choice(self.voice_library["general"]["session_end"]).format(total_reps)
        self.speak(end_msg, priority=9, interrupt=True)
        self.session_active = False

    def clear_queue(self):
        """Remove pending messages"""
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass

    def _stop_current_playback(self):
        """Best-effort stop of current playback"""
        try:
            if self.voice_engine_type == "gtts" and _HAS_PYGAME:
                try:
                    if pygame.mixer.get_init():
                        pygame.mixer.music.stop()
                except Exception:
                    pass
            elif self.voice_engine_type == "pyttsx3" and self.engine:
                try:
                    self.engine.stop()
                except Exception:
                    pass
        except Exception:
            pass

    def _worker(self):
        """Background worker consuming queue and speaking messages"""
        while self._running:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                (_, _), msg = item
                if msg.delay and msg.delay > 0:
                    time.sleep(msg.delay)
                self._speak_text(msg.text)
                self.last_speech_time = time.time()
            except Exception as e:
                print(f"Voice worker error: {e}")
                time.sleep(0.1)

    def _speak_text(self, text: str):
        """Enhanced platform-specific playback with better error handling"""
        try:
            if self.voice_engine_type == "gtts" and _HAS_GTTs and _HAS_PYGAME:
                try:
                    tts = gTTS(text=text, lang="en", slow=False)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        tmp = f.name
                        tts.save(tmp)
                    pygame.mixer.music.load(tmp)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                    try:
                        pygame.mixer.music.unload()
                    except Exception:
                        pass
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"gTTS playback error: {e}; falling back to print.")
                    print(f"[Coach]: {text}")
            elif self.voice_engine_type == "pyttsx3" and _HAS_PYTT and self.engine:
                try:
                    # Check if run loop is already started
                    if hasattr(self, '_run_loop_started') and self._run_loop_started:
                        # If already started, just use say without runAndWait
                        self.engine.say(text)
                        # Don't call runAndWait to avoid the error
                        return
                    else:
                        # First time, use normal flow
                        self.engine.say(text)
                        self.engine.runAndWait()
                        self._run_loop_started = True
                except Exception as e:
                    print(f"pyttsx3 error: {e}; falling back to print.")
                    print(f"[Coach]: {text}")
            else:
                # No TTS available
                print(f"[Coach]: {text}")
        except Exception as e:
            print(f"Speech error: {e}")
            print(f"[Coach]: {text}")

    def shutdown(self, wait: float = 2.0):
        """Enhanced shutdown with better cleanup"""
        self._running = False
        self.clear_queue()
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=wait)
        except Exception:
            pass
        try:
            if self.voice_engine_type == "gtts" and _HAS_PYGAME:
                try:
                    if pygame.mixer.get_init():
                        pygame.mixer.music.stop()
                        pygame.mixer.quit()
                except Exception:
                    pass
            elif self.voice_engine_type == "pyttsx3" and _HAS_PYTT and self.engine:
                try:
                    if hasattr(self, '_run_loop_started'):
                        self.engine.stop()
                except Exception:
                    pass
        except Exception:
            pass

class Rehab360App:
    """Complete Rehab360 application with comprehensive features"""

    def __init__(self):
        """Initialize the complete Rehab360 application"""
        print("🚀 Initializing Rehab360 Pro...")

        # Core components
        self.pose_skeleton = None
        self.angle_visualizer = None
        self.exercise_analyzer = None
        self.voice_coach = None

        # Application state
        self.running = False
        self.current_exercise = "squat"
        self.rep_count = 0
        self.last_rep_time = time.time()

        # Initialize components
        self.initialize_components()

    def initialize_components(self):
        """Initialize all application components with error handling"""
        try:
            # Initialize pose detection and visualization
            self.pose_skeleton = PoseSkeleton()
            print("✅ Pose skeleton initialized")

            # Initialize angle visualizer
            self.angle_visualizer = AngleVisualizer()
            print("✅ Angle visualizer initialized")

            # Initialize exercise analyzer
            self.exercise_analyzer = ExerciseAnalyzer()
            print("✅ Exercise analyzer initialized")

            # Initialize voice coach
            try:
                self.voice_coach = EnhancedVoiceCoach(prefer_gtts=True)
                print("✅ Voice coach initialized")
            except Exception as e:
                print(f"⚠️ Voice coach initialization failed: {e}")
                self.voice_coach = None

            print("✅ Rehab360 Pro initialized successfully!")

        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with comprehensive analysis"""
        try:
            # Detect pose landmarks
            if self.pose_skeleton:
                landmarks = self.pose_skeleton.detect_pose(frame)

                if landmarks:
                    # Draw professional skeleton
                    frame = self.pose_skeleton.draw_professional_skeleton(frame, landmarks)

                    # Calculate angles
                    angles = self.pose_skeleton.get_angles(landmarks)

                    # Draw angles
                    if self.angle_visualizer:
                        frame = self.angle_visualizer.draw_angles(frame, angles, landmarks)

                    # Analyze exercise and provide feedback
                    self.analyze_exercise(angles, landmarks)

            # Add exercise information overlay
            self.draw_exercise_info(frame)

            return frame

        except Exception as e:
            print(f"⚠️ Error processing frame: {e}")
            return frame

    def analyze_exercise(self, angles: Dict[str, float], landmarks: List[Tuple[int, int]]):
        """Analyze current exercise and provide feedback"""
        try:
            if self.exercise_analyzer:
                if self.current_exercise == "squat":
                    result = self.exercise_analyzer.analyze_squat(angles, landmarks)
                elif self.current_exercise == "bicep_curl":
                    result = self.exercise_analyzer.analyze_bicep_curl(angles, landmarks)
                elif self.current_exercise == "shoulder_press":
                    result = self.exercise_analyzer.analyze_shoulder_press(angles, landmarks)
                else:
                    result = {'rep_completed': False, 'form_feedback': [], 'score': 0.8}
            else:
                result = {'rep_completed': False, 'form_feedback': [], 'score': 0.8}

            # Update voice guidance
            if self.voice_coach and result['rep_completed']:
                self.rep_count += 1
                self.voice_coach.guide_exercise(
                    exercise=self.current_exercise,
                    angles=angles,
                    rep_count=self.rep_count,
                    state="completed",
                    form_score=result['score']
                )

                # Provide form feedback
                for feedback in result['form_feedback']:
                    if self.voice_coach:
                        self.voice_coach.speak(feedback, priority=6, category="form_feedback")

        except Exception as e:
            print(f"⚠️ Error analyzing exercise: {e}")

    def draw_exercise_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw exercise information overlay"""
        try:
            # Exercise title
            cv2.putText(frame, f"Exercise: {self.current_exercise.title()}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Rep count
            cv2.putText(frame, f"Reps: {self.rep_count}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Instructions
            instructions = {
                "squat": "Lower until thighs parallel to ground",
                "bicep_curl": "Curl to shoulders, control descent",
                "shoulder_press": "Press overhead, lock out fully"
            }

            if self.current_exercise in instructions:
                cv2.putText(frame, f"Focus: {instructions[self.current_exercise]}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Controls
            controls = "Controls: S=Squat, B=Bicep, P=Press, R=Reset, Q=Quit"
            cv2.putText(frame, controls, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            return frame

        except Exception as e:
            print(f"⚠️ Error drawing exercise info: {e}")
            return frame

    def switch_exercise(self, exercise: str) -> None:
        """Switch to a different exercise with enhanced error handling"""
        try:
            exercise = exercise.lower()
            if exercise not in ["squat", "bicep_curl", "shoulder_press"]:
                print(f"⚠️ Unknown exercise: {exercise}")
                return

            # Update current exercise
            old_exercise = self.current_exercise
            self.current_exercise = exercise
            self.rep_count = 0
            self.last_rep_time = time.time()

            # Reset analyzer
            if self.exercise_analyzer:
                self.exercise_analyzer.reset()

            # Voice transition
            if self.voice_coach:
                try:
                    self.voice_coach.handle_exercise_transition(exercise.title())
                except Exception as e:
                    print(f"⚠️ Voice transition failed: {e}")

            print(f"🔄 Switched from {old_exercise} to {exercise.title()}")

        except Exception as e:
            print(f"Error switching exercise: {e}")

    def run(self) -> None:
        """Main application loop with comprehensive error handling"""
        cap = None
        try:
            # Setup camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam")

            print("✅ Camera setup successful")

            # Welcome message
            if self.voice_coach:
                try:
                    self.voice_coach.speak(
                        "Welcome to Rehab360 Pro. Starting with squats. Get into position and follow my instructions.",
                        priority=9,
                        category="general"
                    )
                except Exception as e:
                    print(f"⚠️ Welcome message failed: {e}")

            self.running = True
            print("🎯 Starting motion analysis...")

            while self.running:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("⚠️ Failed to read frame, retrying...")
                        time.sleep(0.1)
                        continue

                    # Process frame
                    processed_frame = self.process_frame(frame)

                    # Display frame
                    cv2.imshow("Rehab360 Pro - Motion Analysis System", processed_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("👋 Quit requested by user")
                        self.running = False
                    elif key == ord('r'):
                        self.rep_count = 0
                        self.last_rep_time = time.time()
                        if self.exercise_analyzer:
                            self.exercise_analyzer.reset()
                        if self.voice_coach:
                            try:
                                self.voice_coach.speak("Counter reset", priority=7)
                            except Exception as e:
                                print(f"⚠️ Reset message failed: {e}")
                        print("🔄 Counter reset")
                    elif key == ord('s'):
                        self.switch_exercise("squat")
                    elif key == ord('b'):
                        self.switch_exercise("bicep_curl")
                    elif key == ord('p'):
                        self.switch_exercise("shoulder_press")

                except Exception as e:
                    print(f"⚠️ Error in main loop: {e}")
                    time.sleep(0.1)
                    continue

        except Exception as e:
            print(f"❌ Application error: {e}")
        finally:
            self.cleanup(cap)

    def cleanup(self, cap) -> None:
        """Enhanced cleanup with better error handling"""
        print("\n🧹 Cleaning up...")

        try:
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

        # End voice session
        if self.voice_coach:
            try:
                self.voice_coach.end_session(self.rep_count)
                self.voice_coach.shutdown()
            except Exception as e:
                print(f"⚠️ Voice shutdown error: {e}")

        print(f"📊 Final count: {self.rep_count} reps")
        print("👋 Thank you for using Rehab360 Pro!")

def main():
    """Main entry point"""
    try:
        print("🎯 Starting Rehab360 Pro...")
        app = Rehab360App()
        app.run()
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
