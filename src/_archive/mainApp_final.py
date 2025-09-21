"""
Rehab360 Pro - Complete Motion Analysis System
Enhanced with real-time voice coaching and professional skeleton visualization
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import sys
import os
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from voice_coach import EnhancedVoiceCoach
    from pose_draw import PoseSkeleton, AngleVisualizer
    from exercise_detector import ExerciseDetector
    from motion_analyzer import MotionAnalyzer
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    print("Running with basic functionality...")

class Rehab360App:
    """Complete Rehab360 application with comprehensive features"""

    def __init__(self):
        """Initialize the complete Rehab360 application"""
        print("🚀 Initializing Rehab360 Pro...")

        # Core components
        self.pose_detector = None
        self.voice_coach = None
        self.pose_skeleton = None
        self.angle_visualizer = None
        self.exercise_detector = None
        self.motion_analyzer = None

        # Application state
        self.running = False
        self.current_exercise = "squat"
        self.voice_enabled = True
        self.skeleton_enabled = True

        # Initialize components
        self.initialize_components()

    def initialize_components(self):
        """Initialize all application components with error handling"""
        try:
            # Initialize MediaPipe Pose
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ Pose detector initialized")

            # Initialize voice coach
            try:
                self.voice_coach = EnhancedVoiceCoach(prefer_gtts=True)
                print("✅ Voice coach initialized")
            except Exception as e:
                print(f"⚠️ Voice coach initialization failed: {e}")
                self.voice_coach = None

            # Initialize visualization components
            try:
                self.pose_skeleton = PoseSkeleton()
                print("✅ Pose skeleton initialized")
            except Exception as e:
                print(f"⚠️ Pose skeleton initialization failed: {e}")
                self.pose_skeleton = None

            try:
                self.angle_visualizer = AngleVisualizer()
                print("✅ Angle visualizer initialized")
            except Exception as e:
                print(f"⚠️ Angle visualizer initialization failed: {e}")
                self.angle_visualizer = None

            # Basic exercise tracking
            self.rep_count = 0
            self.last_rep_time = time.time()

            print("✅ Rehab360 Pro initialized successfully!")

        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with comprehensive analysis"""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process pose detection
            results = self.pose_detector.process(rgb_frame)

            if results.pose_landmarks:
                # Convert landmarks to pixel coordinates
                h, w = frame.shape[:2]
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))

                # Draw skeleton if enabled
                if self.skeleton_enabled and self.pose_skeleton:
                    frame = self.pose_skeleton.draw_professional_skeleton(frame, landmarks)

                # Basic rep counting for squats
                if self.current_exercise == "squat":
                    self.update_squat_reps(landmarks)

            # Add exercise information overlay
            self.draw_exercise_info(frame)

            return frame

        except Exception as e:
            print(f"⚠️ Error processing frame: {e}")
            return frame

    def update_squat_reps(self, landmarks: List[Tuple[int, int]]):
        """Basic squat rep counting"""
        try:
            if len(landmarks) > 24:  # Ensure we have hip landmarks
                # Simple squat detection based on hip position
                hip_y = landmarks[24][1]  # Right hip
                knee_y = landmarks[26][1]  # Right knee

                current_time = time.time()
                if current_time - self.last_rep_time > 1.0:  # Prevent spam
                    if hip_y > knee_y:  # Squat position
                        self.rep_count += 1
                        self.last_rep_time = current_time
                        print(f"🏋️ Rep {self.rep_count} completed!")

                        # Voice feedback
                        if self.voice_coach and self.voice_enabled:
                            if self.rep_count % 5 == 0:
                                self.voice_coach.speak(f"Great job! {self.rep_count} squats completed!", priority=7)
                            else:
                                self.voice_coach.speak(f"Good rep! That's {self.rep_count}.", priority=6)

        except Exception as e:
            print(f"⚠️ Error updating squat reps: {e}")

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
                "squat": "Stand with feet shoulder-width, lower until thighs parallel",
                "bicep curl": "Keep elbows at sides, curl to shoulders",
                "shoulder press": "Press overhead from shoulder height"
            }

            if self.current_exercise in instructions:
                cv2.putText(frame, f"Instructions: {instructions[self.current_exercise]}",
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
            if exercise not in ["squat", "bicep curl", "shoulder press"]:
                print(f"⚠️ Unknown exercise: {exercise}")
                return

            # Update current exercise
            old_exercise = self.current_exercise
            self.current_exercise = exercise
            self.rep_count = 0
            self.last_rep_time = time.time()

            # Voice transition
            if self.voice_coach and self.voice_enabled:
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
            if self.voice_coach and self.voice_enabled:
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
                        if self.voice_coach and self.voice_enabled:
                            try:
                                self.voice_coach.speak("Counter reset", priority=7)
                            except Exception as e:
                                print(f"⚠️ Reset message failed: {e}")
                        print("🔄 Counter reset")
                    elif key == ord('s'):
                        self.switch_exercise("squat")
                    elif key == ord('b'):
                        self.switch_exercise("bicep curl")
                    elif key == ord('p'):
                        self.switch_exercise("shoulder press")

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
