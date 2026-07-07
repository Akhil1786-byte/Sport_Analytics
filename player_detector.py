# =========================================================
# PLAYER DETECTOR
# =========================================================
from ultralytics import YOLO

class PlayerDetector:

    def __init__(self):
    
        print("[INFO] Loading YOLO model...")
        
        self.model = YOLO("yolov8s.pt")
        print("[INFO] YOLO loaded")

    def detect(self, frame):
        
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.20,
            verbose=False
        )

        return results [0]

        
