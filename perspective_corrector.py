import numpy as np
import cv2


class PerspectiveCorrector:
    

    def __init__(self, H, reference_pixel_point):
        
        self.H = H
        self.reference_jacobian = self._compute_jacobian(reference_pixel_point)

        self.last_pixel_pos = {}  # player_number -> (px, py)

        self.rejected_count = {}  # player_number -> count

    
    MAX_PLAUSIBLE_FRAME_DISTANCE = 1.0

    def _pixel_to_court(self, px, py):
        pt = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)
        return mapped[0][0]

    def _compute_jacobian(self, pixel_point, eps=1.0):
        px, py = pixel_point
        c0 = self._pixel_to_court(px, py)
        cx = self._pixel_to_court(px + eps, py)
        cy = self._pixel_to_court(px, py + eps)
        j1 = (cx - c0) / eps
        j2 = (cy - c0) / eps
        return np.array([j1, j2]).T  # 2x2

    def get_corrected_distance(self, player_number, pixel_pos):
        
        if player_number not in self.last_pixel_pos:
            self.last_pixel_pos[player_number] = pixel_pos
            self.rejected_count.setdefault(player_number, 0)
            return 0.0

        last_px, last_py = self.last_pixel_pos[player_number]
        curr_px, curr_py = pixel_pos

        pixel_delta = np.array([curr_px - last_px, curr_py - last_py])

        corrected_vec = self.reference_jacobian @ pixel_delta
        corrected_distance = float(np.linalg.norm(corrected_vec))

        if corrected_distance > self.MAX_PLAUSIBLE_FRAME_DISTANCE:
            self.rejected_count[player_number] = self.rejected_count.get(player_number, 0) + 1
            return 0.0

        self.last_pixel_pos[player_number] = pixel_pos

        return corrected_distance

    def pixel_to_corrected_point(self, pixel_pos, anchor_court_pos=(0.0, 0.0)):
        
        if not hasattr(self, "_pixel_origin"):
            
            self._pixel_origin = pixel_pos

        origin_px, origin_py = self._pixel_origin
        curr_px, curr_py = pixel_pos

        pixel_delta = np.array([curr_px - origin_px, curr_py - origin_py])
        corrected_offset = self.reference_jacobian @ pixel_delta

        return (
            anchor_court_pos[0] + float(corrected_offset[0]),
            anchor_court_pos[1] + float(corrected_offset[1]),
        )
