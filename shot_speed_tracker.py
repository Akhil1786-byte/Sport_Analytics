import math


class ShotSpeedTracker:
    

    def __init__(self, fps, max_plausible_speed=55.0):
        
        self.fps = fps
        self.max_plausible_speed = max_plausible_speed

        self.last_shot_event = None

        self.player_speeds = {1: [], 2: []}
        self.last_speed_by_player = {1: 0.0, 2: 0.0}

        self.rejected_count = 0  

    def register_shot(self, shot_event):
        
        if self.last_shot_event is None:
            self.last_shot_event = shot_event
            return None

        speed = self._compute_speed(self.last_shot_event, shot_event)
        hit_by = self.last_shot_event.get("hit_by")

        self.last_shot_event = shot_event

        if speed is None:
            self.rejected_count += 1
            return None

        if hit_by in (1, 2):
            self.player_speeds[hit_by].append(speed)
            self.last_speed_by_player[hit_by] = speed

        return {"hit_by": hit_by, "speed": speed}

    def _compute_speed(self, event_a, event_b):
        frame_gap = event_b["frame_idx"] - event_a["frame_idx"]

        if frame_gap <= 0:
            return None  

        elapsed_seconds = frame_gap / self.fps

        pos_a = event_a["court_pos"]
        pos_b = event_b["court_pos"]

        distance = math.sqrt(
            (pos_b[0] - pos_a[0]) ** 2 + (pos_b[1] - pos_a[1]) ** 2
        )

        speed = distance / elapsed_seconds

        if speed > self.max_plausible_speed:
            
            return None

        return speed

    def reset_on_rally_end(self):
        
        self.last_shot_event = None

    def get_last_speed(self, player_number):
        return self.last_speed_by_player.get(player_number, 0.0)

    def get_average_speed(self, player_number):
        speeds = self.player_speeds.get(player_number, [])
        if not speeds:
            return 0.0
        return sum(speeds) / len(speeds)

    def get_max_speed(self, player_number):
        speeds = self.player_speeds.get(player_number, [])
        if not speeds:
            return 0.0
        return max(speeds)

    def get_shot_speed_history(self, player_number):
        return list(self.player_speeds.get(player_number, []))
