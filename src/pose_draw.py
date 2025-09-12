# pose_draw.py
import cv2
import numpy as np
import json

def get_pose_connections():
    """Return complete anatomical connections for full body coverage, based on MediaPipe 33 keypoints."""
    connections = [
        # Head and face (full coverage)
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  # Mouth
        # Neck and torso/spine (connecting shoulders to hips with midline)
        (11, 12), (11, 23), (12, 24), (23, 24),  # Shoulders and hips
        (0, 11), (0, 12),  # Head to shoulders
        (11, 23), (12, 24),  # Spine base
        # Arms (shoulders to fingers with full hand connections)
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (19, 21),  # Left arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (20, 22),  # Right arm
        # Legs and feet (hips to toes with full foot connections)
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # Right leg
        # Enhanced spine/torso (midline and side connections)
        (11, 23), (12, 24), (23, 24),  # Torso base
    ]
    return connections

def draw_glow_skeleton(frame, landmarks, connections, joint_radius=12, inner_radius=6, line_thickness=4, glow_sigma=15, conf_thresh=0.5):
    """Custom drawing with white glowing lines, white circular joints with red centers, Gaussian glow."""
    h, w = frame.shape[:2]
    glow_mask = np.zeros_like(frame, dtype=np.uint8)

    # Draw lines on glow mask for blur effect
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_vis = landmarks[start_idx][2] if len(landmarks[start_idx]) > 2 else 1.0
            end_vis = landmarks[end_idx][2] if len(landmarks[end_idx]) > 2 else 1.0
            if start_vis > conf_thresh and end_vis > conf_thresh:
                start_pt = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                end_pt = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                cv2.line(glow_mask, start_pt, end_pt, (255, 255, 255), line_thickness * 3, cv2.LINE_AA)

    # Draw larger circles on glow mask for joint glow
    for lm in landmarks:
        if len(lm) > 2 and lm[2] > conf_thresh:
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(glow_mask, (x, y), joint_radius * 2, (255, 255, 255), -1, cv2.LINE_AA)

    # Apply Gaussian blur for a stronger glow effect
    glow = cv2.GaussianBlur(glow_mask, (0, 0), sigmaX=glow_sigma, sigmaY=glow_sigma)
    frame = cv2.addWeighted(frame, 0.7, glow, 0.3, 0.0)

    # Draw sharp white lines
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_vis = landmarks[start_idx][2] if len(landmarks[start_idx]) > 2 else 1.0
            end_vis = landmarks[end_idx][2] if len(landmarks[end_idx]) > 2 else 1.0
            if start_vis > conf_thresh and end_vis > conf_thresh:
                start_pt = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                end_pt = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                cv2.line(frame, start_pt, end_pt, (255, 255, 255), line_thickness, cv2.LINE_AA)

    # Draw joints: white outer circle with red center
    for lm in landmarks:
        if len(lm) > 2 and lm[2] > conf_thresh:
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(frame, (x, y), joint_radius, (255, 255, 255), -1, cv2.LINE_AA)  # White outer
            cv2.circle(frame, (x, y), inner_radius, (0, 0, 255), -1, cv2.LINE_AA)  # Red center

    return frame

def draw_angle_gauge(frame, angle, center, label, target_range):
    """Draw an angle gauge for form feedback."""
    cv2.circle(frame, center, 50, (50, 50, 50), -1)
    cv2.circle(frame, center, 48, (255, 255, 255), 2)
    angle_rad = np.radians(angle)
    end_x = center[0] + 40 * np.cos(angle_rad - np.pi / 2)
    end_y = center[1] + 40 * np.sin(angle_rad - np.pi / 2)
    cv2.line(frame, center, (int(end_x), int(end_y)), (0, 200, 0), 3)
    cv2.putText(frame, f"{label}: {int(angle)}°", (center[0] - 40, center[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    min_ang, max_ang = target_range
    if min_ang <= angle <= max_ang:
        cv2.putText(frame, "Good Form", (center[0] - 40, center[1] + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    else:
        cv2.putText(frame, "Adjust Form", (center[0] - 40, center[1] + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def save_landmarks_json(landmarks, frame_index, w, h, path):
    json_data = {
        "frame": frame_index,
        "width": w,
        "height": h,
        "landmarks": [
            {"index": i, "x": lm[0]/w, "y": lm[1]/h, "z": 0.0, "visibility": lm[2] if len(lm) > 2 else 1.0}
            for i, lm in enumerate(landmarks)
        ]
    }
    with open(path, 'w') as f:
        json.dump(json_data, f, indent=4)