import cv2
import numpy as np


class MiniCourt:

    def __init__(self):

        self.width = 240
        self.height = 420
        self.court_width = 8.23
        self.court_length = 23.77

    def _court_to_mini_pixel(self, court_x, court_y):
        cx = max(0, min(court_x, self.court_width))
        cy = max(0, min(court_y, self.court_length))

        px = int(20 + (cx / self.court_width) * (self.width - 40))
        py = int(20 + (cy / self.court_length) * (self.height - 40))

        return px, py

    def draw(
        self,
        frame,
        players,
        distances,
        speeds,
        ball_position=None,
        bounce_markers=None,
        ball_speed=0.0,
        shot_speeds=None,
        avg_speeds=None,
        avg_shot_speeds=None,
    ):
        
        shot_speeds = shot_speeds or {}
        avg_speeds = avg_speeds or {}
        avg_shot_speeds = avg_shot_speeds or {}

        mini = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Court border (doubles width)
        cv2.rectangle(
            mini,
            (20, 20),
            (self.width - 20, self.height - 20),
            (255, 255, 255),
            2
        )

        # Net
        net_y = self.height // 2

        cv2.line(
            mini,
            (20, net_y),
            (self.width - 20, net_y),
            (255, 255, 255),
            2
        )

        # =====================================
        # INNER COURT LINES 
        # =====================================

        playable_width = self.width - 40   
        playable_height = self.height - 40

        DOUBLES_WIDTH_M = 8.23
        SINGLES_WIDTH_M = 6.4
        HALF_COURT_LENGTH_M = 11.885
        SERVICE_LINE_FROM_NET_M = 6.4

        singles_inset_m = (DOUBLES_WIDTH_M - SINGLES_WIDTH_M) / 2
        singles_inset_px = int(singles_inset_m / DOUBLES_WIDTH_M * playable_width)

        left_singles_x = 20 + singles_inset_px
        right_singles_x = (self.width - 20) - singles_inset_px

        
        cv2.line(mini, (left_singles_x, 20), (left_singles_x, self.height - 20), (255, 255, 255), 1)
        cv2.line(mini, (right_singles_x, 20), (right_singles_x, self.height - 20), (255, 255, 255), 1)

        
        service_offset_px = int(
            SERVICE_LINE_FROM_NET_M / HALF_COURT_LENGTH_M * (playable_height / 2)
        )

        far_service_y = net_y - service_offset_px
        near_service_y = net_y + service_offset_px

        cv2.line(mini, (left_singles_x, far_service_y), (right_singles_x, far_service_y), (255, 255, 255), 1)
        cv2.line(mini, (left_singles_x, near_service_y), (right_singles_x, near_service_y), (255, 255, 255), 1)

        
        center_x = (left_singles_x + right_singles_x) // 2
        cv2.line(mini, (center_x, far_service_y), (center_x, near_service_y), (255, 255, 255), 1)

        # =====================================
        # BOUNCE MARKERS 
        # =====================================
        if bounce_markers:
            for marker in bounce_markers:
                court_x, court_y = marker["court_pos"]
                px, py = self._court_to_mini_pixel(court_x, court_y)

                marker_color = (0, 255, 0) if marker["in_bounds"] else (0, 0, 255)

                cv2.circle(mini, (px, py), 5, marker_color, -1)
                cv2.circle(mini, (px, py), 5, (255, 255, 255), 1)

        # Draw Players
        for p in players:

            px, py = self._court_to_mini_pixel(p["court_x"], p["court_y"])

            color = (
                (0, 255, 0)
                if p["player_number"] == 1
                else (0, 0, 255)
            )

            cv2.circle(
                mini,
                (px, py),
                7,
                color,
                -1
            )

            cv2.putText(
                mini,
                str(p["player_number"]),
                (px + 8, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # ==========================
        # DRAW BALL
        # ==========================
        if ball_position is not None:

            px, py = self._court_to_mini_pixel(
                ball_position["court_x"],
                ball_position["court_y"]
            )

            cv2.circle(
                mini,
                (px, py),
                5,
                (0, 255, 255),
                -1
            )

        # Place mini court
        h, w = mini.shape[:2]

        start_x = frame.shape[1] - w - 20
        start_y = 20

        frame[
            start_y:start_y+h,
            start_x:start_x+w
        ] = mini

        # ==================================
        # BALL SPEED BOX 
        # ==================================

        ball_speed_box_y = start_y + h + 10
        ball_speed_box_height = 50

        cv2.rectangle(
            frame,
            (start_x, ball_speed_box_y),
            (start_x + w, ball_speed_box_y + ball_speed_box_height),
            (50, 50, 0),
            -1
        )
        cv2.rectangle(
            frame,
            (start_x, ball_speed_box_y),
            (start_x + w, ball_speed_box_y + ball_speed_box_height),
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Ball Speed: {ball_speed:.2f} m/s",
            (start_x + 10, ball_speed_box_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # ==================================
        # P1 / P2 STATS TABLE 
        # ==================================

        table_y = ball_speed_box_y + ball_speed_box_height + 10

        rows = [
            ("Dist", f"{distances.get(1, 0):.2f}m", f"{distances.get(2, 0):.2f}m"),
            ("Speed", f"{speeds.get(1, 0):.2f}", f"{speeds.get(2, 0):.2f}"),
            ("Shot Spd", f"{shot_speeds.get(1, 0):.2f}", f"{shot_speeds.get(2, 0):.2f}"),
            ("Avg Spd", f"{avg_speeds.get(1, 0):.2f}", f"{avg_speeds.get(2, 0):.2f}"),
            ("Avg Shot", f"{avg_shot_speeds.get(1, 0):.2f}", f"{avg_shot_speeds.get(2, 0):.2f}"),
        ]

        row_height = 28
        label_col_width = 90
        value_col_width = (w - label_col_width) // 2

        header_height = 26
        table_height = header_height + row_height * len(rows)

        
        cv2.rectangle(
            frame,
            (start_x, table_y),
            (start_x + w, table_y + table_height),
            (35, 35, 35),
            -1
        )
        cv2.rectangle(
            frame,
            (start_x, table_y),
            (start_x + w, table_y + table_height),
            (255, 255, 255),
            1
        )

        
        col1_x = start_x + label_col_width
        col2_x = start_x + label_col_width + value_col_width

        cv2.putText(
            frame,
            "P1",
            (col1_x + value_col_width // 2 - 12, table_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            "P2",
            (col2_x + value_col_width // 2 - 12, table_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        
        cv2.line(
            frame,
            (col1_x, table_y),
            (col1_x, table_y + table_height),
            (255, 255, 255),
            1
        )
        cv2.line(
            frame,
            (col2_x, table_y),
            (col2_x, table_y + table_height),
            (255, 255, 255),
            1
        )

        
        cv2.line(
            frame,
            (start_x, table_y + header_height),
            (start_x + w, table_y + header_height),
            (255, 255, 255),
            1
        )

        for i, (label, p1_val, p2_val) in enumerate(rows):
            row_y = table_y + header_height + i * row_height

            cv2.putText(
                frame,
                label,
                (start_x + 8, row_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            cv2.putText(
                frame,
                p1_val,
                (col1_x + 8, row_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            cv2.putText(
                frame,
                p2_val,
                (col2_x + 8, row_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

            if i > 0:
                cv2.line(
                    frame,
                    (start_x, row_y),
                    (start_x + w, row_y),
                    (90, 90, 90),
                    1
                )

        return frame
