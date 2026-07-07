import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np


class CourtLineDetector:

    def __init__(self, model_path):

        self.model = models.resnet50()

        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            14 * 2
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location="cpu"
            )
        )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, frame):

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image_tensor = self.transform(
            rgb
        ).unsqueeze(0)

        with torch.no_grad():

            outputs = self.model(
                image_tensor
            )

        keypoints = (
            outputs
            .squeeze()
            .cpu()
            .numpy()
        )

        keypoints[::2] *= w / 224.0
        keypoints[1::2] *= h / 224.0

        return keypoints

    def draw_keypoints(
        self,
        frame,
        keypoints
    ):

        for i in range(
            0,
            len(keypoints),
            2
        ):

            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            cv2.circle(
                frame,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                str(i // 2),
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame
