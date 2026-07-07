class MatchAnalyticsRecorder:
    

    def __init__(self, fps):
        self.fps = fps

        
        self.samples = {1: [], 2: []}

    def record_frame(self, player_number, frame_idx, distance_delta, speed, court_x=None, court_y=None):
        
        self.samples[player_number].append({
            "frame_idx": frame_idx,
            "seconds": frame_idx / self.fps,
            "distance_delta": distance_delta,
            "speed": speed,
            "court_x": court_x,
            "court_y": court_y,
            "shot_speed": None,
        })

    def record_shot(self, player_number, frame_idx, shot_speed):
        
        player_samples = self.samples[player_number]

        if player_samples and player_samples[-1]["frame_idx"] == frame_idx:
            player_samples[-1]["shot_speed"] = shot_speed
        else:
            player_samples.append({
                "frame_idx": frame_idx,
                "seconds": frame_idx / self.fps,
                "distance_delta": 0.0,
                "speed": 0.0,
                "court_x": None,
                "court_y": None,
                "shot_speed": shot_speed,
            })

    def get_samples(self, player_number):
        return list(self.samples.get(player_number, []))
