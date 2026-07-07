from collections import deque


class BallTracker:
   

    def __init__(self, max_missing=15, history_len=20):

        self.history = deque(maxlen=history_len)

        self.last_ball = None
        self.missing_frames = 0
        self.max_missing = max_missing

    def update(self, detections):
        

        if len(detections) > 0:

            ball = max(detections, key=lambda x: x["conf"])
            center = ball["centre"]

            self.history.append(center)
            self.last_ball = center
            self.missing_frames = 0

            return center, True

        # recovery mode
        self.missing_frames += 1

        if self.last_ball is not None and self.missing_frames < self.max_missing:
            self.history.append(self.last_ball)
            return self.last_ball, False

        return None, False

    def get_history(self):
        return list(self.history)
