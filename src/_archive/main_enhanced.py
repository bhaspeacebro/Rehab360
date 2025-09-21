"""
Rehab360 Pro Enhanced - Real-time Motion Analysis with Comprehensive Voice Coaching
Zero-lag implementation with continuous voice guidance
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import yaml
import sys
import os
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import concurrent.futures

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from pose_draw import PoseSkeleton, AngleVisualizer, ExerciseVisualizer
from voice_coach import RealTimeVoiceCoach

@dataclass
class ExerciseState:
    """Track exercise state for real-time analysis"""
    exercise: str = "squat"
    rep_count: int = 0
    current_phase: str = "neutral"  # neutral, down, bottom, up
    last_phase: str = "neutral"
    form_score: float = 1.0
    angles: Dict[str, float] = None
    rep_in_progress: bool = False
    phase_start_time: float = 0
    
    def __post_init__(self):
        if self.angles is None:
            self.angles = {}

class OptimizedMotionAnalyzer:
    """Optimized motion analysis with predictive processing"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,  # Slightly lower for faster detection
            min_tracking_confidence=0.5,
            model_complexity=1,  # Balanced complexity
            smooth_landmarks=True,
            enable_segmentation=False,  # Disable for performance
            smooth_segmentation=False
        )
        
        # Landmark smoothing
        self.landmark_history = deque(maxlen=3)  # Smaller buffer for responsiveness
        self.last_landmarks = None
        
        # Angle calculation optimization
        self.angle_cache = {}
        self.last_angle_time = 0
        
    def process_frame(self, frame: np.ndarray) -> Optional[List]:
        """Process frame with minimal latency"""
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # Performance optimization
        
        # Process pose
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = [(lm.x, lm.y, lm.z, lm.visibility) 
                        for lm in results.pose_landmarks.landmark]
            
            # Add to history for smoothing
            self.landmark_history.append(landmarks)
            
            # Apply minimal smoothing for responsiveness
            if len(self.landmark_history) >= 2:
                smoothed = self._smooth_landmarks_fast()
                self.last_landmarks = smoothed
                return smoothed
            else:
                self.last_landmarks = landmarks
                return landmarks
        
        return self.last_landmarks
    
    def _smooth_landmarks_fast(self) -> List:
        """Fast landmark smoothing with weighted average"""
        if not self.landmark_history:
            return None
        
        # Weighted average (more weight to recent frames)
        weights = [0.2, 0.3, 0.5] if len(self.landmark_history) == 3 else [0.4, 0.6]
        weights = weights[-len(self.landmark_history):]
        
        smoothed = []
        for i in range(33):  # 33 MediaPipe landmarks
            x_sum = sum(lm[i][0] * w for lm, w in zip(self.landmark_history, weights))
            y_sum = sum(lm[i][1] * w for lm, w in zip(self.landmark_history, weights))
            z_sum = sum(lm[i][2] * w for lm, w in zip(self.landmark_history, weights))
            v_avg = sum(lm[i][3] for lm in self.landmark_history) / len(self.landmark_history)
            smoothed.append((x_sum, y_sum, z_sum, v_avg))
        
        return smoothed
    
    def calculate_angles(self, landmarks: List, exercise: str) -> Dict[str, float]:
        """Calculate exercise-specific angles with caching"""
        if not landmarks or len(landmarks) < 33:
            return {}
        
        current_time = time.time()
        
        # Use cache if very recent (within 50ms)
        if current_time - self.last_angle_time < 0.05:
            return self.angle_cache
        
        angles = {}
        
        try:
            if exercise == "squat":
                # Hip angle
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
                r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                
                angles['hip'] = self._calculate_angle_3d(r_shoulder, r_hip, r_knee)
                angles['knee'] = self._calculate_angle_3d(r_hip, r_knee, r_ankle)
                
            elif exercise == "bicep curl":
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                
                angles['elbow'] = self._calculate_angle_3d(r_shoulder, r_elbow, r_wrist)
                angles['shoulder'] = self._calculate_angle_3d(r_hip, r_shoulder, r_elbow)
                
            elif exercise == "shoulder press":
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                
                angles['elbow'] = self._calculate_angle_3d(r_shoulder, r_elbow, r_wrist)
                angles['shoulder'] = self._calculate_angle_3d(r_hip, r_shoulder, r_elbow)
        
        except (IndexError, TypeError):
            return self.angle_cache
        
        self.angle_cache = angles
        self.last_angle_time = current_time
        
        return angles
    
    def _calculate_angle_3d(self, a, b, c) -> float:
        """Calculate 3D angle between three points"""
        if not all([a, b, c]):
            return 0
        
        # Use only x,y for stability (z can be noisy)
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def cleanup(self):
        """Clean up resources"""
        self.pose.close()

class ExerciseStateDetector:
    """Detect exercise phases with predictive analysis"""
    
    def __init__(self):
        self.phase_thresholds = {
            'squat': {
                'down_start': 150,
                'bottom': 100,
                'up_start': 110,
                'top': 160
            },
            'bicep curl': {
                'up_start': 140,
                'top': 50,
                'down_start': 60,
                'bottom': 150
            },
            'shoulder press': {
                'up_start': 100,
                'top': 160,
                'down_start': 150,
                'bottom': 90
            }
        }
        
        # State prediction
        self.angle_history = deque(maxlen=5)
        self.state_history = deque(maxlen=10)
        
    def detect_phase(self, exercise: str, angles: Dict[str, float], 
                     current_state: ExerciseState) -> str:
        """Detect current exercise phase with prediction"""
        if not angles:
            return current_state.current_phase
        
        # Get primary angle for exercise
        primary_angle = None
        if exercise == "squat" and 'knee' in angles:
            primary_angle = angles['knee']
        elif exercise == "bicep curl" and 'elbow' in angles:
            primary_angle = angles['elbow']
        elif exercise == "shoulder press" and 'elbow' in angles:
            primary_angle = angles['elbow']
        
        if primary_angle is None:
            return current_state.current_phase
        
        # Add to history
        self.angle_history.append(primary_angle)
        
        # Determine phase based on angle and direction
        thresholds = self.phase_thresholds.get(exercise, {})
        new_phase = current_state.current_phase
        
        if exercise == "squat":
            if primary_angle > thresholds['top']:
                new_phase = "ready"
            elif primary_angle < thresholds['bottom']:
                new_phase = "bottom"
            elif current_state.current_phase == "ready" and primary_angle < thresholds['down_start']:
                new_phase = "down"
            elif current_state.current_phase == "bottom" and primary_angle > thresholds['up_start']:
                new_phase = "up"
        
        elif exercise == "bicep curl":
            if primary_angle > thresholds['bottom']:
                new_phase = "ready"
            elif primary_angle < thresholds['top']:
                new_phase = "top"
            elif current_state.current_phase == "ready" and primary_angle < thresholds['up_start']:
                new_phase = "up"
            elif current_state.current_phase == "top" and primary_angle > thresholds['down_start']:
                new_phase = "down"
        
        elif exercise == "shoulder press":
            if primary_angle < thresholds['bottom']:
                new_phase = "ready"
            elif primary_angle > thresholds['top']:
                new_phase = "top"
            elif current_state.current_phase == "ready" and primary_angle > thresholds['up_start']:
                new_phase = "up"
            elif current_state.current_phase == "top" and primary_angle < thresholds['down_start']:
                new_phase = "down"
        
        # Add to state history
        self.state_history.append(new_phase)
        
        return new_phase
    
    def detect_rep_completion(self, current_state: ExerciseState) -> bool:
        """Detect if a rep was completed"""
        if len(self.state_history) < 4:
            return False
        
        # Check for complete cycle
        if current_state.exercise == "squat":
            # Ready -> Down -> Bottom -> Up -> Ready
            if (current_state.last_phase in ["bottom", "up"] and 
                current_state.current_phase == "ready"):
                return True
        
        elif current_state.exercise == "bicep curl":
            # Ready -> Up -> Top -> Down -> Ready
            if (current_state.last_phase in ["top", "down"] and 
                current_state.current_phase == "ready"):
                return True
        
        elif current_state.exercise == "shoulder press":
            # Ready -> Up -> Top -> Down -> Ready
            if (current_state.last_phase in ["top", "down"] and 
                current_state.current_phase == "ready"):
                return True
        
        return False
    
    def calculate_form_score(self, exercise: str, angles: Dict[str, float]) -> float:
        """Calculate form score based on angles"""
        if not angles:
            return 0.5
        
        ideal_angles = {
            'squat': {'knee': 90, 'hip': 90},
            'bicep curl': {'elbow': 45, 'shoulder': 15},
            'shoulder press': {'elbow': 90, 'shoulder': 20}
        }
        
        if exercise not in ideal_angles:
            return 0.5
        
        scores = []
        for joint, ideal in ideal_angles[exercise].items():
            if joint in angles:
                deviation = abs(angles[joint] - ideal)
                score = max(0, 1 - deviation / 90)  # Normalize deviation
                scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.5

class Rehab360ProEnhanced:
    """Enhanced main application with comprehensive voice coaching"""
    
    def __init__(self):
        print("=" * 70)
        print("🏥 REHAB360 PRO ENHANCED - Real-time Motion Analysis")
        print("   With Comprehensive AI Voice Coaching")
        print("=" * 70)
        
        # Initialize components
        self.motion_analyzer = OptimizedMotionAnalyzer()
        self.state_detector = ExerciseStateDetector()
        self.pose_skeleton = PoseSkeleton()
        self.angle_visualizer = AngleVisualizer()
        self.exercise_visualizer = ExerciseVisualizer()
        
        # Initialize voice coach
        print("🎤 Initializing Voice Coach...")
        self.voice_coach = RealTimeVoiceCoach(voice_engine='pyttsx3')
        
        # Exercise state
        self.exercise_state = ExerciseState()
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # UI state
        self.show_skeleton = True
        self.voice_enabled = True
        self.running = True
        
        print("✅ System ready! Stand back and let's begin!")
        print("-" * 70)
    
    def setup_camera(self) -> cv2.VideoCapture:
        """Setup camera with optimal settings"""
        print("📷 Setting up camera...")
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        
        if not cap.isOpened():
            print("❌ ERROR: Camera not found!")
            sys.exit(1)
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Set auto exposure for consistent lighting
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        
        print("✅ Camera ready!")
        return cap
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with all enhancements"""
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get pose landmarks
        landmarks = self.motion_analyzer.process_frame(frame)
        
        if landmarks:
            # Person detected
            if self.exercise_state.current_phase == "not_visible":
                self.voice_coach.handle_person_detected()
            
            # Calculate angles
            angles = self.motion_analyzer.calculate_angles(
                landmarks, self.exercise_state.exercise
            )
            self.exercise_state.angles = angles
            
            # Detect exercise phase
            new_phase = self.state_detector.detect_phase(
                self.exercise_state.exercise, angles, self.exercise_state
            )
            
            # Check for phase transition
            if new_phase != self.exercise_state.current_phase:
                self.exercise_state.last_phase = self.exercise_state.current_phase
                self.exercise_state.current_phase = new_phase
                self.exercise_state.phase_start_time = time.time()
            
            # Check for rep completion
            if self.state_detector.detect_rep_completion(self.exercise_state):
                self.exercise_state.rep_count += 1
                self.exercise_state.rep_in_progress = False
            
            # Calculate form score
            form_score = self.state_detector.calculate_form_score(
                self.exercise_state.exercise, angles
            )
            self.exercise_state.form_score = form_score
            
            # Provide voice guidance
            if self.voice_enabled:
                self.voice_coach.guide_exercise(
                    self.exercise_state.exercise,
                    angles,
                    self.exercise_state.rep_count,
                    self.exercise_state.current_phase,
                    form_score
                )
            
            # Draw skeleton
            if self.show_skeleton:
                # Convert to simple (x,y) format for drawing
                landmarks_2d = [(lm[0], lm[1]) for lm in landmarks]
                frame = self.pose_skeleton.draw_professional_skeleton(frame, landmarks_2d)
            
            # Draw angle gauges
            self._draw_angle_gauges(frame, angles)
            
        else:
            # No person detected
            if self.voice_enabled:
                self.voice_coach.handle_no_person_detected()
            self.exercise_state.current_phase = "not_visible"
        
        # Draw UI overlay
        self._draw_ui(frame)
        
        # Update FPS
        self._update_fps()
        
        return frame
    
    def _draw_angle_gauges(self, frame: np.ndarray, angles: Dict[str, float]):
        """Draw angle visualization gauges"""
        if not angles:
            return
        
        h, w = frame.shape[:2]
        
        positions = {
            'squat': {
                'knee': (w - 120, 180),
                'hip': (w - 120, 300)
            },
            'bicep curl': {
                'elbow': (w - 120, 180),
                'shoulder': (w - 120, 300)
            },
            'shoulder press': {
                'elbow': (w - 120, 180),
                'shoulder': (w - 120, 300)
            }
        }
        
        ideal_ranges = {
            'squat': {'knee': (80, 100), 'hip': (80, 100)},
            'bicep curl': {'elbow': (30, 60), 'shoulder': (0, 30)},
            'shoulder press': {'elbow': (80, 100), 'shoulder': (10, 30)}
        }
        
        if self.exercise_state.exercise in positions:
            ex_positions = positions[self.exercise_state.exercise]
            ex_ranges = ideal_ranges[self.exercise_state.exercise]
            
            for joint, angle in angles.items():
                if joint in ex_positions:
                    self.angle_visualizer.draw_angle_gauge(
                        frame, angle, ex_positions[joint],
                        joint.capitalize(), ex_ranges[joint]
                    )
    
    def _draw_ui(self, frame: np.ndarray):
        """Draw UI overlay with exercise information"""
        h, w = frame.shape[:2]
        
        # Top panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, "REHAB360 PRO", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        # Exercise and reps
        cv2.putText(frame, f"Exercise: {self.exercise_state.exercise.upper()}", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Reps: {self.exercise_state.rep_count}", 
                   (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Form score
        form_color = (0, 255, 0) if self.exercise_state.form_score > 0.7 else (0, 255, 255)
        if self.exercise_state.form_score < 0.5:
            form_color = (0, 0, 255)
        
        cv2.putText(frame, f"Form: {self.exercise_state.form_score:.0%}", 
                   (200, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)
        
        # Phase
        cv2.putText(frame, f"Phase: {self.exercise_state.current_phase.upper()}", 
                   (350, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # FPS
        avg_fps = self._get_avg_fps()
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                   (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Voice status
        voice_status = "Voice: ON" if self.voice_enabled else "Voice: OFF"
        voice_color = (0, 255, 0) if self.voice_enabled else (128, 128, 128)
        cv2.putText(frame, voice_status, 
                   (w - 150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, voice_color, 2)
        
        # Controls
        controls = "S:Squat | B:Bicep | P:Press | V:Voice | K:Skeleton | R:Reset | Q:Quit"
        cv2.putText(frame, controls, 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
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
            self.exercise_state.rep_count = 0
            print("Counter reset!")
        elif key == ord(' '):
            self.print_stats()
    
    def switch_exercise(self, exercise: str):
        """Switch to different exercise"""
        self.exercise_state.exercise = exercise
        self.exercise_state.rep_count = 0
        self.exercise_state.current_phase = "neutral"
        self.exercise_state.last_phase = "neutral"
        print(f"Switched to: {exercise}")
    
    def print_stats(self):
        """Print session statistics"""
        print("\n" + "=" * 50)
        print("📊 SESSION STATS")
        print("-" * 50)
        print(f"Exercise: {self.exercise_state.exercise}")
        print(f"Reps: {self.exercise_state.rep_count}")
        print(f"Form Score: {self.exercise_state.form_score:.0%}")
        print(f"FPS: {self._get_avg_fps():.1f}")
        print("=" * 50 + "\n")
    
    def run(self):
        """Main application loop"""
        cap = self.setup_camera()
        
        # Start voice coaching session
        if self.voice_enabled:
            self.voice_coach.start_session()
        
        print("\n🎯 Application running! Stand in front of the camera...")
        print("Press 'H' for help\n")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera read failed!")
                    break
                
                # Process and display frame
                frame = self.process_frame(frame)
                cv2.imshow("Rehab360 Pro Enhanced - AI Motion Analysis", frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self.handle_keyboard(key)
            
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        
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
            self.voice_coach.end_session(self.exercise_state.rep_count)
            time.sleep(2)  # Let final message play
            self.voice_coach.shutdown()
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.motion_analyzer.cleanup()
        
        # Final stats
        self.print_stats()
        
        print("✅ Thank you for using Rehab360 Pro!")

def main():
    """Main entry point"""
    try:
        app = Rehab360ProEnhanced()
        app.run()
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
