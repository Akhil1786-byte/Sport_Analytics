import math
from collections import defaultdict, deque


class MotionTracker:
    
    DEFAULT_POSITION_ALPHA = {
        1: 0.5,
        2: 0.8,
    }

    
    SPEED_SMOOTHING = 0.5

    
    MAX_PLAUSIBLE_SPEED = 12.0

    def __init__(self, position_alpha_overrides=None):

        self.history = defaultdict(lambda: deque(maxlen=10))
        self.last_speed = defaultdict(float)

        self.smoothed_position = {}  # player_id -> (x, y)

        self.position_alpha = dict(self.DEFAULT_POSITION_ALPHA)
        if position_alpha_overrides:
            self.position_alpha.update(position_alpha_overrides)

        self.rejected_count = defaultdict(int)
        self.last_update_was_rejected = defaultdict(bool)

    def update(self, player_id, court_x, court_y, fps=30):

        alpha = self.position_alpha.get(player_id, 0.6)

        if player_id not in self.smoothed_position:
            self.smoothed_position[player_id] = (court_x, court_y)
            self.history[player_id].append((court_x, court_y))
            return 0.0

        sx, sy = self.smoothed_position[player_id]
        sx = alpha * sx + (1 - alpha) * court_x
        sy = alpha * sy + (1 - alpha) * court_y

        prev_smoothed = self.smoothed_position[player_id]

        dist_check = math.sqrt((sx - prev_smoothed[0]) ** 2 + (sy - prev_smoothed[1]) ** 2)
        speed_check = dist_check * fps

        if speed_check > self.MAX_PLAUSIBLE_SPEED:
            
            self.rejected_count[player_id] += 1
            self.last_update_was_rejected[player_id] = True
            return self.last_speed.get(player_id, 0.0)

        self.last_update_was_rejected[player_id] = False
        self.smoothed_position[player_id] = (sx, sy)
        self.history[player_id].append((sx, sy))

        if len(self.history[player_id]) < 2:
            return 0.0

        prev = self.history[player_id][-2]
        curr = self.history[player_id][-1]

        x1, y1 = prev
        x2, y2 = curr

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speed = dist * fps

        prev_speed = self.last_speed[player_id]
        smooth_speed = self.SPEED_SMOOTHING * prev_speed + (1 - self.SPEED_SMOOTHING) * speed

        self.last_speed[player_id] = smooth_speed

        return smooth_speed

    def get_speed(self, player_id):
        return self.last_speed.get(player_id, 0.0)

    def was_last_update_rejected(self, player_id):
        return self.last_update_was_rejected.get(player_id, False)

    def get_smoothed_position(self, player_id):
        
        return self.smoothed_position.get(player_id)
