"""
Enhanced Pose Detection and Skeleton Visualization
Professional medical-grade skeleton with glowing effects and comprehensive analysis
"""

import cv2
import numpy as np
import mediapipe as mp
import math
from typing import List, Tuple, Optional, Dict, Any
import time

class PoseSkeleton:
    """Professional medical-grade skeleton visualization with glowing effects"""

    def __init__(self):
        # MediaPipe connections for full body skeleton
        self.connections = [
            # Torso and spine
            (11, 12), (11, 23), (12, 24), (23, 24),  # Shoulders and hips

            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),

            # Right arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),

            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),

            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
        ]

        # Enhanced visual configuration for medical-grade appearance
        self.config = {
            "bone_color": (255, 255, 255),      # Pure white bones
            "joint_color": (0, 255, 255),       # Bright cyan joints
            "glow_color": (100, 200, 255),      # Soft blue glow
            "bone_thickness": 4,
            "glow_thickness": 15,
            "joint_radius": 4,      # Reduced from 8 to 4 for smaller joint balls
            "glow_radius": 10,      # Reduced from 25 to 10 for smaller glow
            "opacity": 0.9,
        }

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose  # This is correct usage; mp.solutions.pose exists
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Higher accuracy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Tracking state
        self.prev_landmarks = None
        self.smoothing_factor = 0.3  # Lower smoothing for more responsiveness (was 0.7)
        self.min_visibility = 0.6

    def detect_pose(self, frame: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Detect pose landmarks from frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process pose detection
            results = self.pose_detector.process(rgb_frame)

            if results.pose_landmarks:
                h, w = frame.shape[:2]
                landmarks: List[Tuple[float, float]] = []

                for landmark in results.pose_landmarks.landmark:
                    if landmark.visibility > self.min_visibility:
                        x, y = float(landmark.x * w), float(landmark.y * h)
                        landmarks.append((x, y))
                    else:
                        landmarks.append((-1.0, -1.0))  # Invisible landmark

                # Apply smoothing
                if self.prev_landmarks is not None:
                    landmarks = self._smooth_landmarks(landmarks)

                self.prev_landmarks = landmarks
                return landmarks

            return None

        except Exception as e:
            print(f"Pose detection error: {e}")
            return None

    def _smooth_landmarks(self, current: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply smoothing to reduce jitter"""
        if self.prev_landmarks is None:
            return current

        smoothed = []
        for i, (cx, cy) in enumerate(current):
            if cx == -1.0 or cy == -1.0:
                smoothed.append((-1.0, -1.0))
                continue

            px, py = self.prev_landmarks[i]
            if px == -1.0 or py == -1.0:
                smoothed.append((cx, cy))
                continue

            # Exponential smoothing (lower factor = more responsive)
            sx = px * self.smoothing_factor + cx * (1 - self.smoothing_factor)
            sy = py * self.smoothing_factor + cy * (1 - self.smoothing_factor)
            smoothed.append((sx, sy))

        return smoothed

    def draw_professional_skeleton(self, frame: np.ndarray, landmarks: List[Tuple[float, float]]) -> np.ndarray:
        """Draw professional medical-grade skeleton with enhanced effects"""
        if not landmarks or len(landmarks) < 33:
            return frame

        try:
            h, w = frame.shape[:2]

            # Create overlay layers
            base_layer = np.zeros_like(frame, dtype=np.uint8)
            glow_layer = np.zeros_like(frame, dtype=np.uint8)

            # Convert normalized landmarks to pixel coordinates
            pixel_landmarks = []
            for x, y in landmarks:
                if x >= 0 and y >= 0:  # Valid landmark
                    px = min(max(int(x), 0), w-1)
                    py = min(max(int(y), 0), h-1)
                    pixel_landmarks.append((px, py))
                else:
                    pixel_landmarks.append((-1, -1))

            # Draw skeleton components
            self._draw_glowing_bones(glow_layer, pixel_landmarks)
            self._draw_sharp_bones(base_layer, pixel_landmarks)
            self._draw_enhanced_joints(base_layer, pixel_landmarks)

            # Apply Gaussian blur to glow layer
            glow_layer = cv2.GaussianBlur(glow_layer, (21, 21), 10)

            # Blend layers with the original frame
            result = self._blend_layers(frame, base_layer, glow_layer)

            return result

        except Exception as e:
            print(f"Skeleton drawing error: {e}")
            return frame

    def _draw_glowing_bones(self, layer: np.ndarray, landmarks: List[Tuple[int, int]]) -> None:
        """Draw bones with enhanced glow effect"""
        for start_idx, end_idx in self.connections:
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx] != (-1, -1) and landmarks[end_idx] != (-1, -1)):

                pt1, pt2 = landmarks[start_idx], landmarks[end_idx]
                cv2.line(layer, pt1, pt2,
                        self.config["glow_color"],
                        self.config["glow_thickness"],
                        cv2.LINE_AA)

    def _draw_sharp_bones(self, layer: np.ndarray, landmarks: List[Tuple[int, int]]) -> None:
        """Draw precise bone structure"""
        for start_idx, end_idx in self.connections:
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx] != (-1, -1) and landmarks[end_idx] != (-1, -1)):

                pt1, pt2 = landmarks[start_idx], landmarks[end_idx]
                cv2.line(layer, pt1, pt2,
                        self.config["bone_color"],
                        self.config["bone_thickness"],
                        cv2.LINE_AA)

    def _draw_enhanced_joints(self, layer: np.ndarray, landmarks: List[Tuple[int, int]]) -> None:
        """Draw medical-grade joint visualization"""
        # Get all unique joint indices
        joint_indices = set()
        for start_idx, end_idx in self.connections:
            joint_indices.add(start_idx)
            joint_indices.add(end_idx)

        for idx in joint_indices:
            if idx < len(landmarks) and landmarks[idx] != (-1, -1):
                center = landmarks[idx]

                # Outer glow
                cv2.circle(layer, center,
                          self.config["glow_radius"],
                          self.config["glow_color"], -1, cv2.LINE_AA)

                # Main joint
                cv2.circle(layer, center,
                          self.config["joint_radius"],
                          self.config["joint_color"], -1, cv2.LINE_AA)

                # Center highlight
                cv2.circle(layer, center,
                          self.config["joint_radius"]//2,
                          (255, 255, 255), -1, cv2.LINE_AA)

    def _blend_layers(self, frame: np.ndarray, base_layer: np.ndarray, glow_layer: np.ndarray) -> np.ndarray:
        """Blend skeleton layers with original frame"""
        try:
            # Combine glow and base layers
            combined = cv2.addWeighted(glow_layer, 0.6, base_layer, 0.9, 0)

            # Blend with original frame
            result = cv2.addWeighted(frame, 0.7, combined, 0.8, 0)

            return result

        except Exception as e:
            print(f"Layer blending error: {e}")
            return frame

    def get_angles(self, landmarks: List[Tuple[int, int]]) -> Dict[str, float]:
        """Calculate joint angles for form analysis"""
        angles = {}

        try:
            # Left elbow angle
            if (len(landmarks) > 15 and landmarks[13] != (-1, -1) and
                landmarks[15] != (-1, -1) and landmarks[17] != (-1, -1)):
                angles['left_elbow'] = self._calculate_angle(
                    landmarks[13], landmarks[15], landmarks[17]
                )

            # Right elbow angle
            if (len(landmarks) > 16 and landmarks[14] != (-1, -1) and
                landmarks[16] != (-1, -1) and landmarks[18] != (-1, -1)):
                angles['right_elbow'] = self._calculate_angle(
                    landmarks[14], landmarks[16], landmarks[18]
                )

            # Left knee angle
            if (len(landmarks) > 27 and landmarks[23] != (-1, -1) and
                landmarks[25] != (-1, -1) and landmarks[27] != (-1, -1)):
                angles['left_knee'] = self._calculate_angle(
                    landmarks[23], landmarks[25], landmarks[27]
                )

            # Right knee angle
            if (len(landmarks) > 28 and landmarks[24] != (-1, -1) and
                landmarks[26] != (-1, -1) and landmarks[28] != (-1, -1)):
                angles['right_knee'] = self._calculate_angle(
                    landmarks[24], landmarks[26], landmarks[28]
                )

        except Exception as e:
            print(f"Angle calculation error: {e}")

        return angles

    def _calculate_angle(self, p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
        """Calculate angle between three points"""
        try:
            # Vectors
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            # Dot product
            dot = v1[0] * v2[0] + v1[1] * v2[1]

            # Magnitudes
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

            if mag1 == 0 or mag2 == 0:
                return 0.0

            # Cosine of angle
            cos_angle = dot / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range

            # Convert to degrees
            angle = math.degrees(math.acos(cos_angle))
            return angle

        except Exception as e:
            print(f"Angle calculation error: {e}")
            return 0.0

class AngleVisualizer:
    """Visualize joint angles on the frame"""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.text_color = (255, 255, 255)
        self.bg_color = (0, 0, 0)

    def draw_angles(self, frame: np.ndarray, angles: Dict[str, float],
                   landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Draw angle measurements on frame"""
        try:
            for angle_name, angle_value in angles.items():
                if angle_name == 'left_elbow' and len(landmarks) > 15:
                    pos = landmarks[15]  # Elbow position
                    self._draw_angle_text(frame, pos, f"{angle_value:.1f}°")
                elif angle_name == 'right_elbow' and len(landmarks) > 16:
                    pos = landmarks[16]  # Elbow position
                    self._draw_angle_text(frame, pos, f"{angle_value:.1f}°")
                elif angle_name == 'left_knee' and len(landmarks) > 25:
                    pos = landmarks[25]  # Knee position
                    self._draw_angle_text(frame, pos, f"{angle_value:.1f}°")
                elif angle_name == 'right_knee' and len(landmarks) > 26:
                    pos = landmarks[26]  # Knee position
                    self._draw_angle_text(frame, pos, f"{angle_value:.1f}°")

            return frame

        except Exception as e:
            print(f"Angle visualization error: {e}")
            return frame

    def _draw_angle_text(self, frame: np.ndarray, position: Tuple[int, int], text: str):
        """Draw angle text with background"""
        try:
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                text, self.font, self.font_scale, self.font_thickness
            )

            # Position for text (slightly offset from joint)
            x, y = position
            text_x = x + 15
            text_y = y - 15

            # Draw background rectangle
            cv2.rectangle(frame,
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + baseline + 5),
                         self.bg_color, -1)

            # Draw text
            cv2.putText(frame, text, (text_x, text_y),
                       self.font, self.font_scale, self.text_color, self.font_thickness)

        except Exception as e:
            print(f"Text drawing error: {e}")

# --- Voice Feedback Integration ---
try:
    import pyttsx3
    _voice_engine = pyttsx3.init()
    def speak(text: str):
        _voice_engine.say(text)
        _voice_engine.runAndWait()
except ImportError:
    def speak(text: str):
        print(f"[VOICE FEEDBACK]: {text}")

class ExerciseAnalyzer:
    """Analyze exercise form and provide feedback"""

    def __init__(self):
        self.rep_count = 0
        self.last_rep_time = 0
        self.exercise_state = "ready"
        self.form_score = 0.8

        # Exercise-specific thresholds
        self.thresholds = {
            'squat': {
                'knee_angle_min': 70,
                'knee_angle_max': 130,
                'hip_depth_threshold': 0.7
            },
            'bicep_curl': {
                'elbow_angle_min': 30,
                'elbow_angle_max': 160
            },
            'shoulder_press': {
                'elbow_angle_min': 45,
                'elbow_angle_max': 170
            }
        }

        self.last_feedback = ""
        self.last_feedback_time = 0
        self.feedback_interval = 2.0  # seconds between voice feedback

    def _voice_feedback(self, feedback: str):
        """Speak feedback if it's new or enough time has passed"""
        current_time = time.time()
        if feedback and (feedback != self.last_feedback or current_time - self.last_feedback_time > self.feedback_interval):
            speak(feedback)
            self.last_feedback = feedback
            self.last_feedback_time = current_time

    def analyze_squat(self, angles: Dict[str, float], landmarks: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze squat form"""
        result = {
            'rep_completed': False,
            'form_feedback': [],
            'score': 0.8
        }

        try:
            knee_angles = []
            if 'left_knee' in angles:
                knee_angles.append(angles['left_knee'])
            if 'right_knee' in angles:
                knee_angles.append(angles['right_knee'])

            if knee_angles:
                avg_knee_angle = sum(knee_angles) / len(knee_angles)

                # Squat depth check
                if avg_knee_angle < 80:
                    feedback = "Good depth! Squat lower for better results"
                    result['form_feedback'].append(feedback)
                    result['score'] = 0.9
                elif avg_knee_angle > 120:
                    feedback = "Stand up straighter"
                    result['form_feedback'].append(feedback)
                    result['score'] = 0.6
                else:
                    feedback = "Perfect squat depth!"
                    result['form_feedback'].append(feedback)
                    result['score'] = 1.0

                # Provide voice feedback
                self._voice_feedback(feedback)

                # Rep detection
                current_time = time.time()
                if (self.exercise_state == "up" and avg_knee_angle < 100 and
                    current_time - self.last_rep_time > 1.0):
                    self.rep_count += 1
                    result['rep_completed'] = True
                    self.last_rep_time = current_time
                    self.exercise_state = "down"
                    self._voice_feedback("Rep completed")
                elif avg_knee_angle > 110:
                    self.exercise_state = "up"

        except Exception as e:
            print(f"Squat analysis error: {e}")

        return result

    def analyze_bicep_curl(self, angles: Dict[str, float], landmarks: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze bicep curl form"""
        result = {
            'rep_completed': False,
            'form_feedback': [],
            'score': 0.8
        }

        try:
            elbow_angles = []
            if 'left_elbow' in angles:
                elbow_angles.append(angles['left_elbow'])
            if 'right_elbow' in angles:
                elbow_angles.append(angles['right_elbow'])

            if elbow_angles:
                avg_elbow_angle = sum(elbow_angles) / len(elbow_angles)

                # Range check
                if avg_elbow_angle < 40:
                    feedback = "Full contraction! Squeeze those biceps"
                    result['form_feedback'].append(feedback)
                    result['score'] = 0.9
                elif avg_elbow_angle > 150:
                    feedback = "Lower the weight with control"
                    result['form_feedback'].append(feedback)
                    result['score'] = 0.7
                else:
                    feedback = "Good bicep curl form!"
                    result['form_feedback'].append(feedback)
                    result['score'] = 1.0

                # Provide voice feedback
                self._voice_feedback(feedback)

                # Rep detection
                current_time = time.time()
                if (self.exercise_state == "up" and avg_elbow_angle > 140 and
                    current_time - self.last_rep_time > 0.8):
                    self.rep_count += 1
                    result['rep_completed'] = True
                    self.last_rep_time = current_time
                    self.exercise_state = "down"
                    self._voice_feedback("Rep completed")
                elif avg_elbow_angle < 50:
                    self.exercise_state = "up"

        except Exception as e:
            print(f"Bicep curl analysis error: {e}")

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get current exercise statistics"""
        return {
            'rep_count': self.rep_count,
            'exercise_state': self.exercise_state,
            'form_score': self.form_score,
            'session_time': time.time() - self.last_rep_time if self.last_rep_time > 0 else 0
        }

    def reset(self):
        """Reset exercise tracking"""
        self.rep_count = 0
        self.last_rep_time = time.time()
        self.exercise_state = "ready"
        self.form_score = 0.8
