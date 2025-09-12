# src/pose_tracking/test_cam.py
import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame not captured")
            break
        cv2.imshow("Test Camera", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
