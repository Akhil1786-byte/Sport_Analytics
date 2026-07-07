import numpy as np
from collections import defaultdict


class PlayerTracker:
    

    def __init__(self, homography, H, init_frames=50, relock_after_missing_frames=200):

        self.homography = homography
        self.H = H

        self.init_buffer = []
        self.init_frames = init_frames
        self.locked = False

        self.player1_id = None
        self.player2_id = None

        self.relock_after_missing_frames = relock_after_missing_frames
        self.frames_since_either_seen = 0

        self.track_history = defaultdict(list)

    # =====================================================
    # PIXEL -> COURT
    # =====================================================

    def pixel_to_court(self, pixel):
        try:
            return self.homography.pixel_to_court(pixel, self.H)
        except Exception:
            return (0, 0)

    # =====================================================
    # MAIN
    # =====================================================

    def get_players(self, results):

        if results.boxes is None:
            return self._handle_no_detections()

        frame_detections = self._collect_frame_detections(results)

        if not self.locked:
            return self._run_acquisition_phase(frame_detections)

        return self._run_locked_phase(frame_detections)

    # -----------------------------------------------------------
    # DETECTION COLLECTION
    # -----------------------------------------------------------
    def _collect_frame_detections(self, results):

        frame_detections = []

        for box in results.boxes:

            if box.id is None:
                continue

            cls = int(box.cls[0])
            if results.names[cls] != "person":
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            foot = (center[0], y2)

            
            court_x, court_y = self.pixel_to_court(foot)

            frame_detections.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "center": center,
                "court_x": court_x,
                "court_y": court_y
            })

            self.track_history[track_id].append({
                "center": center,
                "court_x": court_x,
                "court_y": court_y
            })

        return frame_detections

    # -----------------------------------------------------------
    # ACQUIRE LOCK 
    # -----------------------------------------------------------
    def _run_acquisition_phase(self, frame_detections):

        self.init_buffer.append(frame_detections)

        if len(self.init_buffer) < self.init_frames:
            return []  # still collecting

        movement_score = defaultdict(float)
        top_side = defaultdict(int)
        bottom_side = defaultdict(int)

        for frame in self.init_buffer:
            for det in frame:

                tid = det["track_id"]
                y = det["court_y"]

                hist = self.track_history[tid]
                if len(hist) > 1:
                    prev = hist[-2]["center"]
                    curr = hist[-1]["center"]
                    movement_score[tid] += np.linalg.norm(np.array(curr) - np.array(prev))

                if y < 11.88:
                    top_side[tid] += 1
                else:
                    bottom_side[tid] += 1

        valid_ids = [tid for tid in movement_score if movement_score[tid] > 30]

        if len(valid_ids) < 2:
            
            if len(self.init_buffer) > self.init_frames * 3:
                self.init_buffer = self.init_buffer[-self.init_frames:]
            return []

        top_candidates = [tid for tid in valid_ids if top_side[tid] > bottom_side[tid]]
        bottom_candidates = [tid for tid in valid_ids if bottom_side[tid] > top_side[tid]]

        if len(top_candidates) > 0 and len(bottom_candidates) > 0:

            self.player2_id = max(top_candidates, key=lambda x: movement_score[x])
            self.player1_id = max(bottom_candidates, key=lambda x: movement_score[x])

            self.locked = True
            self.frames_since_either_seen = 0
            self.init_buffer = []  # free the buffer, no longer needed until next re-lock

            print(f"[LOCKED] P1={self.player1_id}, P2={self.player2_id}")

        return []

    # -----------------------------------------------------------
    # PHASE 3: NORMAL TRACKING AFTER LOCK, WITH RE-LOCK 
    # -----------------------------------------------------------
    def _run_locked_phase(self, frame_detections):

        players = []
        seen_any_locked_id = False

        for det in frame_detections:

            if det["track_id"] == self.player1_id:
                players.append({"player_number": 1, "bbox": det["bbox"]})
                seen_any_locked_id = True

            elif det["track_id"] == self.player2_id:
                players.append({"player_number": 2, "bbox": det["bbox"]})
                seen_any_locked_id = True

        if seen_any_locked_id:
            self.frames_since_either_seen = 0
        else:
            self.frames_since_either_seen += 1

        if self.frames_since_either_seen >= self.relock_after_missing_frames:
            
            print(
                f"[RELOCK] Lost P1={self.player1_id}, P2={self.player2_id} for "
                f"{self.frames_since_either_seen} frames -- re-acquiring."
            )
            self._reset_for_relock()
            return []

        return players

    def _handle_no_detections(self):

        if self.locked:
            self.frames_since_either_seen += 1

            if self.frames_since_either_seen >= self.relock_after_missing_frames:
                print("[RELOCK] No detections at all for too long -- re-acquiring.")
                self._reset_for_relock()

        return []

    def _reset_for_relock(self):

        self.locked = False
        self.player1_id = None
        self.player2_id = None
        self.init_buffer = []
        self.frames_since_either_seen = 0
