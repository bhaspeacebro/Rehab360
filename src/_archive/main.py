"""
Rehab360 Pro - Advanced Real-time Motion Analysis System
With AI-Powered Physiotherapist and Optimized Performance
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import json
import yaml
import sys
import os
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import concurrent.futures
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from pose_draw import PoseSkeleton, AngleVisualizer, ExerciseVisualizer, smooth_landmarks
from ai_physiotherapist import AIPhysiotherapist, PhysiotherapistFeedback, MovementPattern

# Try to import performance optimizations
try:
    import numba
    from numba import jit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    print("Numba not available. Running without JIT optimization.")

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "app_config.yaml"
        
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found at {self.config_path}. Using defaults.")
            return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'camera': {
                'device_id': 0,
                'width': 1280,
                'height': 720,
                'fps': 30,
                'flip_horizontal': True
            },
            'pose_detection': {
                'model_complexity': 1,
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.7,
                'smooth_landmarks': True,
                'smoothing_window': 5
            },
            'physiotherapist': {
                'enabled': True,
                'personality': 'encouraging',
                'voice_engine': 'gtts'
            },
            'performance': {
                'use_gpu': True,
                'max_fps': 30,
                'async_processing': True
            },
            'ui': {
                'show_fps': True,
                'show_angles': True,
                'show_rep_count': True
            }
        }

@dataclass
class PerformanceMetrics:
    """Track performance metrics for optimization"""
    fps: float = 0.0
    frame_time: float = 0.0
    pose_detection_time: float = 0.0
    drawing_time: float = 0.0
    ai_processing_time: float = 0.0
    total_frames: int = 0
    dropped_frames: int = 0

class OptimizedPoseProcessor:
    """Optimized pose processing with frame skipping and caching"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=config['pose_detection']['min_detection_confidence'],
            min_tracking_confidence=config['pose_detection']['min_tracking_confidence'],
            model_complexity=config['pose_detection']['model_complexity']
        )
        
        # Performance optimization
        self.frame_skip = config.get('performance', {}).get('frame_skip', 0)
        self.frame_counter = 0
        self.last_landmarks = None
        self.landmark_cache = deque(maxlen=10)
        
        # Threading for async processing
        self.use_async = config.get('performance', {}).get('async_processing', True)
        if self.use_async:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            self.pending_future = None
    
    def process_frame(self, frame: np.ndarray) -> Optional[List]:
        """Process frame with optimizations"""
        self.frame_counter += 1
        
        # Frame skipping for performance
        if self.frame_skip > 0 and self.frame_counter % (self.frame_skip + 1) != 0:
            return self.last_landmarks
        
        # Async processing
        if self.use_async:
            return self._process_async(frame)
        else:
            return self._process_sync(frame)
    
    def _process_sync(self, frame: np.ndarray) -> Optional[List]:
        """Synchronous processing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
            self.last_landmarks = landmarks
            self.landmark_cache.append(landmarks)
            return landmarks
        
        return self.last_landmarks
    
    def _process_async(self, frame: np.ndarray) -> Optional[List]:
        """Asynchronous processing for better performance"""
        # Check if previous async operation completed
        if self.pending_future and self.pending_future.done():
            try:
                landmarks = self.pending_future.result(timeout=0.01)
                if landmarks:
                    self.last_landmarks = landmarks
                    self.landmark_cache.append(landmarks)
            except:
                pass
            self.pending_future = None
        
        # Start new async operation if none pending
        if not self.pending_future:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.pending_future = self.executor.submit(self._process_frame_async, rgb_frame)
        
        return self.last_landmarks
    
    def _process_frame_async(self, rgb_frame: np.ndarray) -> Optional[List]:
        """Async frame processing worker"""
        results = self.pose.process(rgb_frame)
        if results.pose_landmarks:
            return [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        return None
    
    def get_smoothed_landmarks(self) -> Optional[List]:
        """Get smoothed landmarks from cache"""
        if len(self.landmark_cache) >= 3:
            return smooth_landmarks(list(self.landmark_cache))
        elif self.last_landmarks:
            return self.last_landmarks
        return None
    
    def cleanup(self):
        """Clean up resources"""
        self.pose.close()
        if self.use_async and hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

class ExerciseAnalyzer:
    """Optimized exercise analysis with caching"""
    
    def __init__(self, mp_pose):
        self.mp_pose = mp_pose
        self.angle_cache = {}
        self.last_analysis_time = 0
        self.analysis_cooldown = 0.1  # Minimum time between analyses
    
    @staticmethod
    def calculate_angle_optimized(a: Tuple[float, float], 
                                 b: Tuple[float, float], 
                                 c: Tuple[float, float]) -> float:
        """Optimized angle calculation"""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        # Fast dot product and norm calculation
        dot_product = np.dot(ba, bc)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba * norm_bc == 0:
            return 0
        
        cosine_angle = dot_product / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def analyze_squat(self, landmarks: List) -> Dict[str, float]:
        """Analyze squat with caching"""
        current_time = time.time()
        
        # Use cache if recent
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return self.angle_cache.get('squat', {})
        
        try:
            # Get key landmarks
            r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            angles = {
                'hip': self.calculate_angle_optimized(r_shoulder, r_hip, r_knee),
                'knee': self.calculate_angle_optimized(r_hip, r_knee, r_ankle)
            }
            
            self.angle_cache['squat'] = angles
            self.last_analysis_time = current_time
            
            return angles
            
        except (IndexError, TypeError):
            return self.angle_cache.get('squat', {})
    
    def analyze_bicep_curl(self, landmarks: List) -> Dict[str, float]:
        """Analyze bicep curl with caching"""
        current_time = time.time()
        
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return self.angle_cache.get('bicep_curl', {})
        
        try:
            r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            angles = {
                'elbow': self.calculate_angle_optimized(r_shoulder, r_elbow, r_wrist),
                'shoulder': self.calculate_angle_optimized(r_hip, r_shoulder, r_elbow)
            }
            
            self.angle_cache['bicep_curl'] = angles
            self.last_analysis_time = current_time
            
            return angles
            
        except (IndexError, TypeError):
            return self.angle_cache.get('bicep_curl', {})
    
    def analyze_shoulder_press(self, landmarks: List) -> Dict[str, float]:
        """Analyze shoulder press with caching"""
        current_time = time.time()
        
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return self.angle_cache.get('shoulder_press', {})
        
        try:
            r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            angles = {
                'elbow': self.calculate_angle_optimized(r_shoulder, r_elbow, r_wrist),
                'shoulder': self.calculate_angle_optimized(r_hip, r_shoulder, r_elbow)
            }
            
            self.angle_cache['shoulder_press'] = angles
            self.last_analysis_time = current_time
            
            return angles
            
        except (IndexError, TypeError):
            return self.angle_cache.get('shoulder_press', {})

class FPSCalculator:
    """Calculate FPS with smoothing"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self):
        """Update FPS calculation"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        self.last_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS"""
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time == 0:
            return 0.0
        
        return 1.0 / avg_frame_time

class Rehab360ProApp:
    """Main application class with optimized performance"""
    
    def __init__(self, config_path: Optional[str] = None):
        print("🚀 Initializing Rehab360 Pro - Advanced Motion Analysis System")
        print("=" * 60)
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        
        # Initialize components with optimizations
        self.pose_processor = OptimizedPoseProcessor(self.config)
        self.pose_skeleton = PoseSkeleton()
        self.angle_visualizer = AngleVisualizer()
        self.exercise_visualizer = ExerciseVisualizer()
        self.exercise_analyzer = ExerciseAnalyzer(self.mp_pose)
        
        # Initialize AI Physiotherapist
        if self.config['physiotherapist']['enabled']:
            print("🤖 Initializing AI Physiotherapist...")
            self.physiotherapist = AIPhysiotherapist()
            self.physiotherapist.personality = self.config['physiotherapist']['personality']
        else:
            self.physiotherapist = None
        
        # Exercise tracking
        self.current_exercise = "squat"
        self.rep_count = 0
        self.last_state = "neutral"
        self.form_errors = 0
        self.session_start_time = time.time()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.fps_calculator = FPSCalculator()
        
        # Application state
        self.running = True
        self.show_skeleton = True
        self.voice_enabled = self.config['physiotherapist']['enabled']
        
        print("✅ System initialization complete!")
        print(f"📊 Configuration: {self.config['camera']['width']}x{self.config['camera']['height']} @ {self.config['camera']['fps']} FPS")
        print(f"🎯 AI Physiotherapist: {'Enabled' if self.voice_enabled else 'Disabled'}")
        print("-" * 60)
    
    def setup_camera(self) -> cv2.VideoCapture:
        """Setup camera with optimal settings"""
        print("📷 Configuring camera...")
        
        cap = cv2.VideoCapture(self.config['camera']['device_id'])
        
        if not cap.isOpened():
            print("❌ ERROR: Could not open camera")
            sys.exit(1)
        
        # Set camera properties from config
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Set additional properties for performance
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        print("✅ Camera configured successfully")
        return cap
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """Process single frame with all optimizations"""
        start_time = time.time()
        
        # Flip frame if configured
        if self.config['camera']['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        
        # Pose detection
        pose_start = time.time()
        landmarks = self.pose_processor.process_frame(frame)
        self.metrics.pose_detection_time = time.time() - pose_start
        
        feedback_data = None
        
        if landmarks:
            # Get smoothed landmarks
            if self.config['pose_detection']['smooth_landmarks']:
                smoothed_landmarks = self.pose_processor.get_smoothed_landmarks()
                if smoothed_landmarks:
                    landmarks = smoothed_landmarks
            
            # Analyze exercise
            angles = {}
            if self.current_exercise == "squat":
                angles = self.exercise_analyzer.analyze_squat(landmarks)
            elif self.current_exercise == "bicep curl":
                angles = self.exercise_analyzer.analyze_bicep_curl(landmarks)
            elif self.current_exercise == "shoulder press":
                angles = self.exercise_analyzer.analyze_shoulder_press(landmarks)
            
            # Track repetitions
            self._update_rep_count(angles)
            
            # Generate AI feedback
            if self.physiotherapist and self.voice_enabled and angles:
                ai_start = time.time()
                feedback = self.physiotherapist.analyze_movement(
                    self.current_exercise,
                    angles,
                    landmarks,
                    self.rep_count
                )
                
                if feedback and feedback.text:
                    self.physiotherapist.speak(feedback)
                    feedback_data = {
                        'text': feedback.text,
                        'category': feedback.category,
                        'angles': angles
                    }
                
                self.metrics.ai_processing_time = time.time() - ai_start
            
            # Draw visualization
            draw_start = time.time()
            
            if self.show_skeleton:
                frame = self.pose_skeleton.draw_professional_skeleton(frame, landmarks)
            
            # Draw angle gauges
            if self.config['ui']['show_angles'] and angles:
                self._draw_angle_gauges(frame, angles)
            
            self.metrics.drawing_time = time.time() - draw_start
        
        # Draw UI overlay
        self._draw_ui_overlay(frame, feedback_data)
        
        # Update metrics
        self.metrics.frame_time = time.time() - start_time
        self.metrics.total_frames += 1
        
        return frame, feedback_data
    
    def _update_rep_count(self, angles: Dict[str, float]):
        """Update repetition count based on angles"""
        if not angles:
            return
        
        # Determine state based on exercise
        current_state = "neutral"
        
        if self.current_exercise == "squat":
            if 'knee' in angles:
                if angles['knee'] > 160:
                    current_state = "up"
                elif angles['knee'] < 110:
                    current_state = "down"
        
        elif self.current_exercise == "bicep curl":
            if 'elbow' in angles:
                if angles['elbow'] > 150:
                    current_state = "down"
                elif angles['elbow'] < 50:
                    current_state = "up"
        
        elif self.current_exercise == "shoulder press":
            if 'elbow' in angles:
                if angles['elbow'] > 160:
                    current_state = "up"
                elif angles['elbow'] < 90:
                    current_state = "down"
        
        # Count reps on state transitions
        if self.last_state == "down" and current_state == "up":
            self.rep_count += 1
        
        self.last_state = current_state
    
    def _draw_angle_gauges(self, frame: np.ndarray, angles: Dict[str, float]):
        """Draw angle visualization gauges"""
        h, w = frame.shape[:2]
        
        gauge_positions = {
            'squat': {
                'knee': (w - 100, 150),
                'hip': (w - 100, 250)
            },
            'bicep curl': {
                'elbow': (w - 100, 150),
                'shoulder': (w - 100, 250)
            },
            'shoulder press': {
                'elbow': (w - 100, 150),
                'shoulder': (w - 100, 250)
            }
        }
        
        target_ranges = {
            'squat': {'knee': (80, 110), 'hip': (140, 190)},
            'bicep curl': {'elbow': (40, 80), 'shoulder': (0, 30)},
            'shoulder press': {'elbow': (150, 170), 'shoulder': (10, 40)}
        }
        
        if self.current_exercise in gauge_positions:
            positions = gauge_positions[self.current_exercise]
            ranges = target_ranges[self.current_exercise]
            
            for joint, angle in angles.items():
                if joint in positions:
                    self.angle_visualizer.draw_angle_gauge(
                        frame, angle, positions[joint], 
                        joint.capitalize(), ranges[joint]
                    )
    
    def _draw_ui_overlay(self, frame: np.ndarray, feedback_data: Optional[Dict]):
        """Draw UI overlay with exercise information"""
        h, w = frame.shape[:2]
        
        # Top panel
        cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)
        
        # Exercise info
        cv2.putText(frame, f"EXERCISE: {self.current_exercise.upper()}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Rep count
        cv2.putText(frame, f"REPS: {self.rep_count}", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Form errors
        cv2.putText(frame, f"ERRORS: {self.form_errors}", 
                   (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                   (0, 0, 255) if self.form_errors > 0 else (0, 255, 0), 2)
        
        # FPS
        fps = self.fps_calculator.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # AI Status
        ai_status = "AI: ON" if self.voice_enabled else "AI: OFF"
        color = (0, 255, 0) if self.voice_enabled else (128, 128, 128)
        cv2.putText(frame, ai_status, 
                   (w - 150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Feedback display
        if feedback_data and feedback_data.get('text'):
            # Create semi-transparent background for feedback
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, h - 80), (w - 10, h - 10), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Display feedback text
            feedback_text = feedback_data['text']
            cv2.putText(frame, feedback_text[:100], 
                       (20, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Instructions
        instructions = "S:Squat | B:Bicep | P:Press | R:Reset | V:Voice | K:Skeleton | Q:Quit"
        cv2.putText(frame, instructions, 
                   (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.rep_count = 0
            self.form_errors = 0
            print("✅ Counter reset")
        elif key == ord('s'):
            self.switch_exercise("squat")
        elif key == ord('b'):
            self.switch_exercise("bicep curl")
        elif key == ord('p'):
            self.switch_exercise("shoulder press")
        elif key == ord('v'):
            self.voice_enabled = not self.voice_enabled
            status = "enabled" if self.voice_enabled else "disabled"
            print(f"🔊 Voice feedback {status}")
        elif key == ord('k'):
            self.show_skeleton = not self.show_skeleton
            status = "visible" if self.show_skeleton else "hidden"
            print(f"💀 Skeleton overlay {status}")
        elif key == ord(' '):
            self.print_session_stats()
        
        return True
    
    def switch_exercise(self, exercise: str):
        """Switch to different exercise"""
        self.current_exercise = exercise
        self.rep_count = 0
        self.last_state = "neutral"
        
        print(f"🔄 Switched to {exercise}")
        
        if self.physiotherapist and self.voice_enabled:
            self.physiotherapist.speak(
                PhysiotherapistFeedback(
                    text=f"Let's do some {exercise} exercises. Get into position!",
                    priority=5,
                    category='instruction',
                    emotion='encouraging'
                )
            )
    
    def print_session_stats(self):
        """Print current session statistics"""
        session_time = time.time() - self.session_start_time
        
        print("\n" + "=" * 60)
        print("📊 SESSION STATISTICS")
        print("-" * 60)
        print(f"Exercise: {self.current_exercise}")
        print(f"Total Reps: {self.rep_count}")
        print(f"Form Errors: {self.form_errors}")
        print(f"Session Time: {session_time:.1f} seconds")
        print(f"Average FPS: {self.fps_calculator.get_fps():.1f}")
        
        if self.rep_count > 0:
            accuracy = ((self.rep_count - self.form_errors) / self.rep_count) * 100
            print(f"Form Accuracy: {accuracy:.1f}%")
            print(f"Reps/Minute: {(self.rep_count / session_time * 60):.1f}")
        
        print("=" * 60 + "\n")
    
    def run(self):
        """Main application loop"""
        cap = self.setup_camera()
        
        # Welcome message
        print("\n🎯 Starting workout session...")
        print("Press 'H' for help with controls\n")
        
        if self.physiotherapist and self.voice_enabled:
            self.physiotherapist.speak(
                PhysiotherapistFeedback(
                    text="Welcome to Rehab360 Pro! I'm your AI physiotherapist. Let's start with squats!",
                    priority=5,
                    category='instruction',
                    emotion='encouraging'
                ),
                force=True
            )
        
        try:
            while self.running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                # Process frame
                frame, feedback = self.process_frame(frame)
                
                # Update FPS
                self.fps_calculator.update()
                
                # Display frame
                cv2.imshow("Rehab360 Pro - AI Motion Analysis", frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_keyboard(key):
                        break
        
        except KeyboardInterrupt:
            print("\n⏹ Session interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup(cap)
    
    def cleanup(self, cap: cv2.VideoCapture):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        # Final statistics
        self.print_session_stats()
        
        # AI physiotherapist summary
        if self.physiotherapist:
            if self.voice_enabled:
                self.physiotherapist.speak(
                    PhysiotherapistFeedback(
                        text="Great workout! Keep up the excellent work!",
                        priority=5,
                        category='motivation',
                        emotion='encouraging'
                    ),
                    force=True
                )
                
                # Wait for final speech
                time.sleep(2)
            
            self.physiotherapist.shutdown()
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.pose_processor.cleanup()
        
        print("✅ Cleanup complete. Thank you for using Rehab360 Pro!")

def main():
    """Main entry point"""
    try:
        app = Rehab360ProApp()
        app.run()
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
