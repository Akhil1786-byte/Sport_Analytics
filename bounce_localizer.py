class BounceLocalizer:
    
    COURT_WIDTH = 8.23
    COURT_LENGTH = 23.77
    OUT_OF_BOUNDS_MARGIN = 0.5

    def __init__(self, max_recent=10, display_lifetime_frames=45):
        self.max_recent = max_recent
        self.display_lifetime_frames = display_lifetime_frames

        self.recent_bounces = []  
        self.all_bounces = []     

    def register_bounce(self, bounce_event, current_frame_idx):
        
        court_x, court_y = bounce_event["court_pos"]

        side = "far" if court_y < 11.88 else "near"
        in_bounds = self._is_in_bounds(court_x, court_y)

        localized = {
            "frame_idx": bounce_event["frame_idx"],
            "pixel_pos": bounce_event["pixel_pos"],
            "court_pos": bounce_event["court_pos"],
            "side": side,
            "in_bounds": in_bounds,
            "expires_at_frame": current_frame_idx + self.display_lifetime_frames,
        }

        self.recent_bounces.append(localized)
        self.all_bounces.append(localized)

        if len(self.recent_bounces) > self.max_recent:
            self.recent_bounces.pop(0)

        return localized

    def _is_in_bounds(self, court_x, court_y):
        m = self.OUT_OF_BOUNDS_MARGIN
        return (
            -m <= court_x <= self.COURT_WIDTH + m
            and -m <= court_y <= self.COURT_LENGTH + m
        )

    def get_active_markers(self, current_frame_idx):
        
        self.recent_bounces = [
            b for b in self.recent_bounces if b["expires_at_frame"] >= current_frame_idx
        ]
        return list(self.recent_bounces)

    def get_all_bounces(self):
        return list(self.all_bounces)

    def get_bounce_counts_by_side(self):
        counts = {"near": 0, "far": 0}
        for b in self.all_bounces:
            counts[b["side"]] += 1
        return counts

    def get_in_out_counts(self):
        counts = {"in": 0, "out": 0}
        for b in self.all_bounces:
            counts["in" if b["in_bounds"] else "out"] += 1
        return counts
