# src/mainApp.py

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import threading
from collections import deque
import json
import math

print("⏱️ Starting Rehab360…\n")

# ---------------- Voice Assistant -----------------
class VoiceAssistant:
    def __init__(self, rate=150, cooldown=2.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)  # female if available
        self.last_spoken = 0.0
        self.cooldown = cooldown
        self.queue = []
        self.is_speaking = False

    def say(self, text, priority=0):
        self.queue.append((text, priority, time.time()))
        self.queue.sort(key=lambda x: x[1], reverse=True)
        self.process_queue()

    def process_queue(self):
        if self.is_speaking or not self.queue:
            return
        now = time.time()
        if now - self.last_spoken < self.cooldown:
            return

        text, priority, _ = self.queue.pop(0)
        self.last_spoken = now
        self.is_speaking = True

        def speak():
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_speaking = False
            self.process_queue()

        threading.Thread(target=speak).start()

# ---------------- Exercise Tracker -----------------
class ExerciseTracker:
    def __init__(self):
        self.rep_count = 0
        self.state = "up"
        self.last_state = "up"
        self.rep_start_time = 0
        self.rep_durations = []
        self.form_errors = 0

    def reset(self):
        self.rep_count = 0
        self.state = "up"
        self.last_state = "up"
        self.rep_durations = []
        self.form_errors = 0

    def update_rep_count(self, new_state):
        if self.last_state == "up" and new_state == "down":
            self.rep_start_time = time.time()
        elif self.last_state == "down" and new_state == "up":
            self.rep_count += 1
            rep_time = time.time() - self.rep_start_time
            self.rep_durations.append(rep_time)
        self.last_state = self.state
        self.state = new_state

    def get_stats(self):
        if not self.rep_durations:
            return {"avg_duration": 0, "total_reps": 0, "form_errors": self.form_errors}
        return {
            "avg_duration": sum(self.rep_durations) / len(self.rep_durations),
            "total_reps": self.rep_count,
            "form_errors": self.form_errors
        }

# ---------------- Utility -----------------
def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def draw_angle_gauge(frame, angle, pos, label, good_range):
    """Draw circular gauge with angle value"""
    x, y = pos
    radius = 40

    # Determine color based on good range
    if good_range[0] <= angle <= good_range[1]:
        color = (0, 255, 0)  # Green
    elif angle < good_range[0]:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red

    # Draw outer circle
    cv2.circle(frame, (x, y), radius, (200, 200, 200), 2)

    # Draw arc showing angle progress
    end_angle = int((angle / 180) * 360)
    cv2.ellipse(frame, (x, y), (radius, radius), -90, 0, end_angle, color, 4)

    # Text label + value
    cv2.putText(frame, f"{label}: {int(angle)}°", (x - 45, y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ---------------- Feedback Logic -----------------
def get_squat_feedback(hip_ang, knee_ang, tracker):
    if knee_ang < 110:
        tracker.update_rep_count("down")
    elif knee_ang > 160:
        tracker.update_rep_count("up")

    if knee_ang > 160:
        return "Stand tall", 1
    elif knee_ang < 70:
        tracker.form_errors += 1
        return "Too deep! Stop lower", 2
    elif 80 <= knee_ang <= 110:
        return "Good squat depth", 1
    else:
        return "Go deeper", 1

def get_bicep_curl_feedback(elbow_ang, shoulder_ang, tracker):
    if elbow_ang < 50:
        tracker.update_rep_count("down")
    elif elbow_ang > 150:
        tracker.update_rep_count("up")

    if elbow_ang > 150:
        return "Fully extend arm", 1
    elif elbow_ang < 50:
        return "Good curl!", 1
    elif shoulder_ang > 40:
        tracker.form_errors += 1
        return "Keep shoulder steady", 2
    else:
        return "Keep curling", 1

def get_shoulder_press_feedback(elbow_ang, shoulder_ang, tracker):
    if elbow_ang > 160:
        tracker.update_rep_count("up")
    elif elbow_ang < 90:
        tracker.update_rep_count("down")

    if elbow_ang < 90:
        return "Lower arms to start", 1
    elif elbow_ang > 160:
        return "Good press up!", 1
    elif shoulder_ang > 50:
        tracker.form_errors += 1
        return "Don't shrug shoulders", 2
    else:
        return "Press higher", 1

# ---------------- Main -----------------
def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Camera not found")
        return

    buffer = deque(maxlen=7)  # smoothing
    voice = VoiceAssistant(rate=150, cooldown=2.0)
    tracker = ExerciseTracker()

    current_exercise = "squat"
    voice.say("Starting squat exercise. Get into position.")

    try:
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                cv2.putText(frame, f"Exercise: {current_exercise.title()}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                cv2.putText(frame, f"Reps: {tracker.rep_count}", (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    landmarks = [(lm.x, lm.y) for lm in result.pose_landmarks.landmark]
                    buffer.append(landmarks)

                    avg_landmarks = []
                    for i in range(len(buffer[0])):
                        xs = [lm[i][0] for lm in buffer]
                        ys = [lm[i][1] for lm in buffer]
                        avg_landmarks.append((np.mean(xs), np.mean(ys)))

                    feedback, priority = "", 0
                    if current_exercise == "squat":
                        rhip = avg_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                        rknee = avg_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                        rankle = avg_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                        rshoulder = avg_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                        hip_ang = calculate_angle(rshoulder, rhip, rknee)
                        knee_ang = calculate_angle(rhip, rknee, rankle)
                        feedback, priority = get_squat_feedback(hip_ang, knee_ang, tracker)

                        draw_angle_gauge(frame, knee_ang, (w - 100, 100), "Knee", (80, 110))
                        draw_angle_gauge(frame, hip_ang, (w - 220, 100), "Hip", (140, 190))

                    elif current_exercise == "bicep curl":
                        rshoulder = avg_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        relbow = avg_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                        rwrist = avg_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                        rhip = avg_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                        elbow_ang = calculate_angle(rshoulder, relbow, rwrist)
                        shoulder_ang = calculate_angle(rhip, rshoulder, relbow)
                        feedback, priority = get_bicep_curl_feedback(elbow_ang, shoulder_ang, tracker)

                        draw_angle_gauge(frame, elbow_ang, (w - 100, 100), "Elbow", (50, 80))
                        draw_angle_gauge(frame, shoulder_ang, (w - 220, 100), "Shoulder", (10, 30))

                    elif current_exercise == "shoulder press":
                        rshoulder = avg_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        relbow = avg_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                        rwrist = avg_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                        rhip = avg_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                        elbow_ang = calculate_angle(rshoulder, relbow, rwrist)
                        shoulder_ang = calculate_angle(rhip, rshoulder, relbow)
                        feedback, priority = get_shoulder_press_feedback(elbow_ang, shoulder_ang, tracker)

                        draw_angle_gauge(frame, elbow_ang, (w - 100, 100), "Elbow", (150, 170))
                        draw_angle_gauge(frame, shoulder_ang, (w - 220, 100), "Shoulder", (10, 30))

                    cv2.putText(frame, feedback, (20, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
                    if feedback and priority > 0:
                        voice.say(feedback, priority)

                cv2.imshow("Rehab360", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    tracker.reset()
                    voice.say("Counter reset")
                elif key == ord('s'):
                    tracker.reset(); current_exercise = "squat"; voice.say("Squat mode")
                elif key == ord('b'):
                    tracker.reset(); current_exercise = "bicep curl"; voice.say("Bicep curl mode")
                elif key == ord('p'):
                    tracker.reset(); current_exercise = "shoulder press"; voice.say("Shoulder press mode")

    finally:
        stats = tracker.get_stats()
        print("\nWorkout Summary:", json.dumps(stats, indent=2))
        voice.say(f"Workout complete. {stats['total_reps']} reps done.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
