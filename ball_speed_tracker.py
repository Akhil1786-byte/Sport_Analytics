import math
from collections import deque


class BallSpeedTracker:
    

    def __init__(self, max_speed=80.0, history_len=5):

        self.prev_point = None
        self.frames_since_prev = 0

        self.speed_history = deque(maxlen=history_len)

        self.max_speed = max_speed
        self.rejected_count = 0

    def update(self, point, fps, is_real_detection=True):

        if point is None or not is_real_detection:
            self.frames_since_prev += 1
            return self.speed_history[-1] if len(self.speed_history) > 0 else 0.0

        if self.prev_point is None:
            self.prev_point = point
            self.frames_since_prev = 0
            return 0.0

        # ==================================
        # DISTANCE / TIME
        # ==================================

        dx = point[0] - self.prev_point[0]
        dy = point[1] - self.prev_point[1]

        distance = math.sqrt(dx * dx + dy * dy)

        frame_gap = max(1, self.frames_since_prev + 1)
        elapsed_seconds = frame_gap / fps

        speed = distance / elapsed_seconds

        # ==================================
        # SPIKE HANDLING -- REJECT, DON'T CLIP
        # ==================================

        if speed > self.max_speed:
            self.rejected_count += 1
            
            self.frames_since_prev += 1
            return self.speed_history[-1] if len(self.speed_history) > 0 else 0.0

        # ==================================
        # SMOOTHING
        # ==================================

        self.speed_history.append(speed)

        smoothed_speed = sum(self.speed_history) / len(self.speed_history)

        self.prev_point = point
        self.frames_since_prev = 0

        return smoothed_speed
