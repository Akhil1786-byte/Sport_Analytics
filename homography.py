import cv2
import numpy as np


class CourtHomography:

    def __init__(self):

        self.court_points = np.array([
            [0.00, 0.00],
            [8.23, 0.00],
            [0.00, 23.77],
            [8.23, 23.77],

            [1.37, 0.00],
            [6.86, 0.00],
            [1.37, 23.77],
            [6.86, 23.77],

            [0.00, 11.885],
            [8.23, 11.885],

            [1.37, 11.885],
            [6.86, 11.885],

            [4.115, 0.00],
            [4.115, 23.77]
        ], dtype=np.float32)

    def compute(self, keypoints):

        image_points = []

        for i in range(14):

            x = keypoints[i * 2]
            y = keypoints[i * 2 + 1]

            image_points.append([x, y])

        image_points = np.array(
            image_points,
            dtype=np.float32
        )

        H, _ = cv2.findHomography(
            image_points,
            self.court_points,
            cv2.RANSAC
        )

        return H

    def pixel_to_court(self, pixel, H):

        pt = np.array(
            [[[pixel[0], pixel[1]]]],
            dtype=np.float32
        )

        mapped = cv2.perspectiveTransform(
            pt,
            H
        )

        return (
            float(mapped[0][0][0]),
            float(mapped[0][0][1])
        )
