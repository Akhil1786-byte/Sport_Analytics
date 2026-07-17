import math

import cv2
import mediapipe as mp


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


class PlayerPoseEstimator:
    

    def __init__(self, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def get_keypoints(self, frame, bbox, padding=15):
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        x1p = max(0, int(x1) - padding)
        y1p = max(0, int(y1) - padding)
        x2p = min(w, int(x2) + padding)
        y2p = min(h, int(y2) + padding)

        if x2p <= x1p or y2p <= y1p:
            return None

        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(crop_rgb)

        if not results.pose_landmarks:
            return None

        crop_h, crop_w = crop.shape[:2]
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            px = x1p + lm.x * crop_w
            py = y1p + lm.y * crop_h
            keypoints.append((px, py, lm.visibility))

        return keypoints

    def draw_keypoints(self, frame, keypoints, color=(0, 255, 255), min_visibility=0.5):
        
        if keypoints is None:
            return frame

        for start_idx, end_idx in self.mp_pose.POSE_CONNECTIONS:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue

            x1, y1, v1 = keypoints[start_idx]
            x2, y2, v2 = keypoints[end_idx]

            if v1 < min_visibility or v2 < min_visibility:
                continue

            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        for (x, y, v) in keypoints:
            if v < min_visibility:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

        return frame

    def get_elbow_angles(self, keypoints, min_visibility=0.5):
        
        if keypoints is None:
            return {"left_elbow_angle": None, "right_elbow_angle": None}

        left_angle = self._angle_between(
            keypoints, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, min_visibility
        )
        right_angle = self._angle_between(
            keypoints, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, min_visibility
        )

        return {"left_elbow_angle": left_angle, "right_elbow_angle": right_angle}

    @staticmethod
    def _angle_between(keypoints, idx_a, idx_b, idx_c, min_visibility):
        
        if idx_a >= len(keypoints) or idx_b >= len(keypoints) or idx_c >= len(keypoints):
            return None

        ax, ay, av = keypoints[idx_a]
        bx, by, bv = keypoints[idx_b]
        cx, cy, cv = keypoints[idx_c]

        if av < min_visibility or bv < min_visibility or cv < min_visibility:
            return None

        
        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)

        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)

        if mag1 == 0 or mag2 == 0:
            return None

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))  # clamp for float safety

        return math.degrees(math.acos(cos_angle))

    def close(self):
        self.pose.close()
