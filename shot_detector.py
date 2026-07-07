from collections import deque
import math


class ShotDetector:

    def __init__(
        self,
        proximity_threshold=2.5,
        speed_threshold=4.0,
        speed_multiplier=1.5,
        cooldown_frames=10
    ):

        self.proximity_threshold = proximity_threshold
        self.speed_threshold = speed_threshold
        self.speed_multiplier = speed_multiplier
        self.cooldown_frames = cooldown_frames

        self.frame_idx = 0
        self.last_shot_frame = -100

        self.prev_ball_speed = 0.0

        self.shot_events = []

    def update(
        self,
        ball_court_pos,
        ball_speed,
        player_court_positions
    ):

        self.frame_idx += 1

        if ball_court_pos is None:
            self.prev_ball_speed = ball_speed
            return None

        if self.frame_idx - self.last_shot_frame < self.cooldown_frames:
            self.prev_ball_speed = ball_speed
            return None

        speed_spike = (
            ball_speed > self.speed_threshold and
            ball_speed > self.prev_ball_speed * self.speed_multiplier
        )

        if not speed_spike:
            self.prev_ball_speed = ball_speed
            return None

        nearest_player = None
        nearest_dist = float("inf")

        for player_num, player_pos in player_court_positions.items():

            if player_pos is None:
                continue

            dist = math.sqrt(
                (ball_court_pos[0] - player_pos[0]) ** 2 +
                (ball_court_pos[1] - player_pos[1]) ** 2
            )

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_player = player_num

        if nearest_dist > self.proximity_threshold:
            self.prev_ball_speed = ball_speed
            return None

        shot_event = {
            "frame_idx": self.frame_idx,
            "hit_by": nearest_player,
            "court_pos": ball_court_pos,
            "ball_speed": ball_speed
        }

        self.shot_events.append(shot_event)

        self.last_shot_frame = self.frame_idx

        print(
            f"[SHOT] "
            f"P{nearest_player} "
            f"speed={ball_speed:.2f} "
            f"dist={nearest_dist:.2f}"
        )

        self.prev_ball_speed = ball_speed

        return shot_event

    def get_shot_events(self):
        return self.shot_events
