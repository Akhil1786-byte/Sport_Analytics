from ultralytics import YOLO

class BallDetector:

    def __init__(self,
                model_path="models/yolo5_last.onnx"):

        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model.predict(
            frame,
            conf=0.10,
            verbose=False        
        )[0]

        balls = []

        for box in results.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            conf = float(box.conf[0])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            balls.append({
                "bbox": [x1, y1,x2, y2],
                "centre": (cx, cy),
                "conf": conf
            })

        return balls
