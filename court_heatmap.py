import numpy as np
import cv2


class CourtHeatmapGenerator:
    

    COURT_WIDTH = 8.23
    COURT_LENGTH = 23.77

    def __init__(self, grid_cols=10, grid_rows=20, image_width=400, image_height=900):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.image_width = image_width
        self.image_height = image_height

    def _build_grid(self, samples):
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)

        for s in samples:
            cx, cy = s.get("court_x"), s.get("court_y")
            if cx is None or cy is None:
                continue

            
            cx = max(0.0, min(cx, self.COURT_WIDTH))
            cy = max(0.0, min(cy, self.COURT_LENGTH))

            col = min(self.grid_cols - 1, int((cx / self.COURT_WIDTH) * self.grid_cols))
            row = min(self.grid_rows - 1, int((cy / self.COURT_LENGTH) * self.grid_rows))

            grid[row, col] += 1

        return grid

    def get_dominant_zone(self, samples):
        
        grid = self._build_grid(samples)

        if grid.sum() == 0:
            return "insufficient position data"

        row, col = np.unravel_index(np.argmax(grid), grid.shape)

        
        depth_fraction = row / self.grid_rows
        if depth_fraction < 0.35:
            depth_desc = "far baseline area"
        elif depth_fraction < 0.65:
            depth_desc = "mid-court / net area"
        else:
            depth_desc = "near baseline area"

        
        width_fraction = col / self.grid_cols
        if width_fraction < 0.35:
            width_desc = "left side"
        elif width_fraction < 0.65:
            width_desc = "center"
        else:
            width_desc = "right side"

        time_in_zone_pct = 100.0 * grid[row, col] / grid.sum()

        return f"{depth_desc}, {width_desc} ({time_in_zone_pct:.1f}% of recorded time in this cell)"

    def render_heatmap_image(self, samples, player_label="Player"):
        
        grid = self._build_grid(samples)

        img = np.full((self.image_height, self.image_width, 3), 255, dtype=np.uint8)

        margin = 30
        court_px_width = self.image_width - 2 * margin
        court_px_height = self.image_height - 2 * margin

        if grid.sum() > 0:
            normalized = grid / grid.max()
        else:
            normalized = grid

        cell_w = court_px_width / self.grid_cols
        cell_h = court_px_height / self.grid_rows

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                intensity = normalized[row, col]
                if intensity <= 0:
                    continue

                
                color = self._intensity_to_color(intensity)

                x1 = int(margin + col * cell_w)
                y1 = int(margin + row * cell_h)
                x2 = int(margin + (col + 1) * cell_w)
                y2 = int(margin + (row + 1) * cell_h)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        
        cv2.rectangle(img, (margin, margin), (self.image_width - margin, self.image_height - margin), (0, 0, 0), 2)

        net_y = margin + court_px_height // 2
        cv2.line(img, (margin, net_y), (self.image_width - margin, net_y), (0, 0, 0), 2)

        
        singles_inset_m = (8.23 - 6.4) / 2
        singles_inset_px = int((singles_inset_m / self.COURT_WIDTH) * court_px_width)
        cv2.line(img, (margin + singles_inset_px, margin), (margin + singles_inset_px, self.image_height - margin), (120, 120, 120), 1)
        cv2.line(img, (self.image_width - margin - singles_inset_px, margin), (self.image_width - margin - singles_inset_px, self.image_height - margin), (120, 120, 120), 1)

        cv2.putText(img, f"{player_label} Court Coverage Heatmap", (margin, margin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return img

    @staticmethod
    def _intensity_to_color(intensity):
        
        intensity = max(0.0, min(1.0, intensity))

        if intensity < 0.5:
            t = intensity / 0.5
            b = int(255 * (1 - t))
            g = int(255 * t)
            r = 0
        else:
            t = (intensity - 0.5) / 0.5
            b = 0
            g = int(255 * (1 - t))
            r = int(255 * t)

        return (b, g, r)

    def save_heatmap(self, samples, filepath, player_label="Player"):
        img = self.render_heatmap_image(samples, player_label=player_label)
        cv2.imwrite(filepath, img)
        return filepath
