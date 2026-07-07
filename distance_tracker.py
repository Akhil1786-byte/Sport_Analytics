import math


class DistanceTracker:
    

    DEFAULT_JITTER_THRESHOLD = {
        1: 0.01,
        2: 0.02,  
    }

    def __init__(self, jitter_threshold_overrides=None):

        self.total = {1: 0.0, 2: 0.0}
        self.speed = {1: 0.0, 2: 0.0}

        self.jitter_threshold = dict(self.DEFAULT_JITTER_THRESHOLD)
        if jitter_threshold_overrides:
            self.jitter_threshold.update(jitter_threshold_overrides)

    def update(self, pid, corrected_distance_increment):
        
        threshold = self.jitter_threshold.get(pid, 0.01)

        if corrected_distance_increment < threshold:
            return

        self.total[pid] += corrected_distance_increment

    def update_speed(self, pid, s):
        self.speed[pid] = s

    def get_distance(self, pid):
        return self.total.get(pid, 0.0)

    def get_speed(self, pid):
        return self.speed.get(pid, 0.0)
