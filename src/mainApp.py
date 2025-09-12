# mainApp.py
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import threading
from collections import deque
import json
import math
import os

from pose_draw import get_pose_connections, draw_glow_skeleton, draw_angle_gauge, save_landmarks_json

print("⏱ Starting Rehab360…\n")

class VoiceAssistant:
    def __init__(self, rate=150, cooldown=2.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
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

        threading.Thread(target=speak, daemon=True).start()

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

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

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

def extract_landmarks(result, frame_width, frame_height):
    landmarks = []
    if result.pose_landmarks:
        for idx, lm in enumerate(result.pose_landmarks.landmark):
            landmarks.append((
                lm.x * frame_width,
                lm.y * frame_height,
                lm.visibility
            ))
    return landmarks

def main():
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ ERROR: Camera not found")
        return

    buffer = deque(maxlen=3)  # Minimal buffering
    voice = VoiceAssistant(rate=150, cooldown=2.0)
    tracker = ExerciseTracker()

    current_exercise = "squat"
    voice.say("Starting squat exercise. Get into position.")

    frame_index = 0
    EMA_ALPHA = 0.3  # Smoothing factor
    prev_smoothed_landmarks = None

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    pose_connections = get_pose_connections()

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
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
                    landmarks = extract_landmarks(result, w, h)

                    buffer.append(landmarks)

                    if len(buffer) == buffer.maxlen:
                        avg_landmarks = np.mean(buffer, axis=0).tolist()

                        if prev_smoothed_landmarks is None:
                            smoothed_landmarks = avg_landmarks
                        else:
                            smoothed_landmarks = []
                            for prev, curr in zip(prev_smoothed_landmarks, avg_landmarks):
                                x = EMA_ALPHA * curr[0] + (1 - EMA_ALPHA) * prev[0]
                                y = EMA_ALPHA * curr[1] + (1 - EMA_ALPHA) * prev[1]
                                v = EMA_ALPHA * curr[2] + (1 - EMA_ALPHA) * prev[2]
                                smoothed_landmarks.append((x, y, v))
                        prev_smoothed_landmarks = smoothed_landmarks

                        frame = draw_glow_skeleton(frame, smoothed_landmarks, pose_connections)

                        feedback, priority = "", 0
                        if current_exercise == "squat":
                            rhip = smoothed_landmarks[24]
                            rknee = smoothed_landmarks[26]
                            rankle = smoothed_landmarks[28]
                            rshoulder = smoothed_landmarks[12]

                            hip_ang = calculate_angle(rshoulder, rhip, rknee)
                            knee_ang = calculate_angle(rhip, rknee, rankle)
                            feedback, priority = get_squat_feedback(hip_ang, knee_ang, tracker)

                            draw_angle_gauge(frame, knee_ang, (w - 100, 100), "Knee", (80, 110))
                            draw_angle_gauge(frame, hip_ang, (w - 220, 100), "Hip", (140, 190))

                        elif current_exercise == "bicep curl":
                            rshoulder = smoothed_landmarks[12]
                            relbow = smoothed_landmarks[14]
                            rwrist = smoothed_landmarks[16]
                            rhip = smoothed_landmarks[24]

                            elbow_ang = calculate_angle(rshoulder, relbow, rwrist)
                            shoulder_ang = calculate_angle(rhip, rshoulder, relbow)
                            feedback, priority = get_bicep_curl_feedback(elbow_ang, shoulder_ang, tracker)

                            draw_angle_gauge(frame, elbow_ang, (w - 100, 100), "Elbow", (50, 80))
                            draw_angle_gauge(frame, shoulder_ang, (w - 220, 100), "Shoulder", (10, 30))

                        elif current_exercise == "shoulder press":
                            rshoulder = smoothed_landmarks[12]
                            relbow = smoothed_landmarks[14]
                            rwrist = smoothed_landmarks[16]
                            rhip = smoothed_landmarks[24]

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
                    tracker.reset()
                    current_exercise = "squat"
                    voice.say("Squat mode")
                elif key == ord('b'):
                    tracker.reset()
                    current_exercise = "bicep curl"
                    voice.say("Bicep curl mode")
                elif key == ord('p'):
                    tracker.reset()
                    current_exercise = "shoulder press"
                    voice.say("Shoulder press mode")
                elif key == ord('c'):
                    png_path = f"outputs/output_{frame_index}.png"
                    json_path = f"models/pose_landmarks_{frame_index}.json"
                    cv2.imwrite(png_path, frame)
                    landmarks = extract_landmarks(result, w, h)
                    save_landmarks_json(landmarks, frame_index, w, h, json_path)
                    save_landmarks_json(landmarks, frame_index, w, h, "models/pose_landmarks_last.json")
                    print(f"Saved PNG: {png_path}")
                    print(f"Saved JSON: {json_path}")

                frame_index += 1

    finally:
        stats = tracker.get_stats()
        print("\nWorkout Summary:", json.dumps(stats, indent=2))
        voice.say(f"Workout complete. {stats['total_reps']} reps done.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()