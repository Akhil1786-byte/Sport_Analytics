import math
from collections import deque

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


class CatBoostBounceDetector:
    

    NUM_LAGS = 3
    WINDOW_SIZE = 5

    def __init__(self, model_path, threshold=0.45, max_gap_to_interpolate=3):
        import catboost as ctb

        self.model = ctb.CatBoostRegressor()
        self.model.load_model(model_path)
        self.threshold = threshold
        self.max_gap_to_interpolate = max_gap_to_interpolate

        self.buffer = deque(maxlen=self.WINDOW_SIZE + 2)

        self.frame_idx = -1

        self.cooldown_frames = 10
        self.frames_since_last_bounce = self.cooldown_frames

        self.bounce_events = []

    def update(self, ball_pixel_pos, is_real_detection):
        
        self.frame_idx += 1
        self.frames_since_last_bounce += 1

        x, y = (None, None)
        if is_real_detection and ball_pixel_pos is not None:
            x, y = ball_pixel_pos

        self.buffer.append({"frame_idx": self.frame_idx, "x": x, "y": y})

        if len(self.buffer) < self.WINDOW_SIZE:
            return None

        window = list(self.buffer)[-self.WINDOW_SIZE:]
        center_pos = self.WINDOW_SIZE // 2

        xs = [w["x"] for w in window]
        ys = [w["y"] for w in window]

        xs, ys = self._fill_small_gaps(xs, ys)

        if any(v is None for v in xs) or any(v is None for v in ys):
            return None

        if self.frames_since_last_bounce < self.cooldown_frames:
            return None

        features = self._compute_features(xs, ys, center_pos)
        if features is None:
            return None

        pred = self.model.predict([features])[0]

        if pred <= self.threshold:
            return None

        center_frame = window[center_pos]
        event = {
            "frame_idx": center_frame["frame_idx"],
            "pixel_pos": (xs[center_pos], ys[center_pos]),
            "score": float(pred),
        }

        self.bounce_events.append(event)
        self.frames_since_last_bounce = 0

        return event

    def _fill_small_gaps(self, xs, ys):
        xs = list(xs)
        ys = list(ys)

        none_count = sum(1 for v in xs if v is None)
        if none_count == 0:
            return xs, ys
        if none_count > self.max_gap_to_interpolate:
            return xs, ys

        known_idx = [i for i, v in enumerate(xs) if v is not None]
        if len(known_idx) < 2:
            return xs, ys

        for i in range(len(xs)):
            if xs[i] is not None:
                continue

            before = [j for j in known_idx if j < i]
            after = [j for j in known_idx if j > i]

            if len(before) >= 2:
                src_idx = before[-2:]
            elif len(before) == 1 and after:
                src_idx = before + after[:1]
            elif len(after) >= 2:
                src_idx = after[:2]
            else:
                continue

            try:
                rel_xs = list(range(len(src_idx)))
                cs_x = CubicSpline(rel_xs, [xs[j] for j in src_idx], bc_type="natural") \
                    if len(src_idx) > 2 else None
                if cs_x is not None:
                    xs[i] = float(cs_x(len(src_idx)))
                    ys[i] = float(CubicSpline(rel_xs, [ys[j] for j in src_idx], bc_type="natural")(len(src_idx)))
                else:
                    x0, x1 = xs[src_idx[0]], xs[src_idx[1]]
                    y0, y1 = ys[src_idx[0]], ys[src_idx[1]]
                    xs[i] = x0 + (x1 - x0)
                    ys[i] = y0 + (y1 - y0)
            except Exception:
                continue

        return xs, ys

    def _compute_features(self, xs, ys, center_pos):
        eps = 1e-15
        cx, cy = xs[center_pos], ys[center_pos]

        feats = {}

        for i in range(1, self.NUM_LAGS):
            x_lag = xs[center_pos - i]
            x_lag_inv = xs[center_pos + i]
            y_lag = ys[center_pos - i]
            y_lag_inv = ys[center_pos + i]

            x_diff = abs(x_lag - cx)
            y_diff = y_lag - cy
            x_diff_inv = abs(x_lag_inv - cx)
            y_diff_inv = y_lag_inv - cy

            x_div = abs(x_diff / (x_diff_inv + eps))
            y_div = y_diff / (y_diff_inv + eps)

            feats[f"x_diff_{i}"] = x_diff
            feats[f"x_diff_inv_{i}"] = x_diff_inv
            feats[f"x_div_{i}"] = x_div
            feats[f"y_diff_{i}"] = y_diff
            feats[f"y_diff_inv_{i}"] = y_diff_inv
            feats[f"y_div_{i}"] = y_div

        colnames_x = [f"x_diff_{i}" for i in range(1, self.NUM_LAGS)] + \
                     [f"x_diff_inv_{i}" for i in range(1, self.NUM_LAGS)] + \
                     [f"x_div_{i}" for i in range(1, self.NUM_LAGS)]
        colnames_y = [f"y_diff_{i}" for i in range(1, self.NUM_LAGS)] + \
                     [f"y_diff_inv_{i}" for i in range(1, self.NUM_LAGS)] + \
                     [f"y_div_{i}" for i in range(1, self.NUM_LAGS)]

        ordered_cols = colnames_x + colnames_y

        return [feats[c] for c in ordered_cols]

    def get_bounce_events(self):
        return list(self.bounce_events)
