import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

print("⏱ Starting Rehab360…\n")

# ==============================
# Voice Assistant for Feedback
# ==============================
class VoiceAssistant:
    def __init__(self, rate=150, cooldown=2.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.last_spoken_time = 0
        self.cooldown = cooldown  # seconds

    def speak(self, text):
        """Speaks only if cooldown time has passed"""
        current_time = time.time()
        if current_time - self.last_spoken_time >= self.cooldown:
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_spoken_time = current_time


# ==============================
# Exercise Tracker
# ==============================
class ExerciseTracker:
    def __init__(self):
        self.count = 0
        self.stage = None  # "up" or "down"
        self.feedback = ""

    def calculate_angle(self, a, b, c):
        """Calculates angle between three points"""
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def update_bicep_curl(self, landmarks):
        """Update count and feedback for bicep curls"""
        mp_pose = mp.solutions.pose
        # Get coordinates
        shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
        ]
        wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
        ]

        # Calculate angle
        angle = self.calculate_angle(shoulder, elbow, wrist)

        # Curl counter logic
        if angle > 160:
            self.stage = "down"
        if angle < 40 and self.stage == "down":
            self.stage = "up"
            self.count += 1
            self.feedback = f"Good! Rep {self.count} completed"
        elif angle > 90 and self.stage == "up":
            self.feedback = "Keep your elbow tucked in"


# ==============================
# Draw skeleton (better than single line)
# ==============================
def draw_styled_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )


# ==============================
# Main App
# ==============================
def main():
    cap = cv2.VideoCapture(0)

    # Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize
    tracker = ExerciseTracker()
    assistant = VoiceAssistant()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        if results.pose_landmarks:
            tracker.update_bicep_curl(results.pose_landmarks.landmark)
            draw_styled_landmarks(image, results)

            # Voice feedback
            if tracker.feedback:
                assistant.speak(tracker.feedback)

        # Render info
        cv2.rectangle(image, (0, 0), (250, 100), (245, 117, 16), -1)

        cv2.putText(image, "REPS", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, str(tracker.count), (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        cv2.putText(image, "STAGE", (120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, tracker.stage if tracker.stage else "-", (120, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        cv2.imshow("Rehab360", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
