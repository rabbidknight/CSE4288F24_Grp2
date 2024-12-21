# Load trained model
from ultralytics import YOLO

trained_model = YOLO('../../OneDrive/Desktop/YOLO-crosswalk/runs/detect/train4/weights/best.pt')  # Adjust path

# Run inference
results = trained_model.predict(source="test3.jpg", save=True, imgsz=1024, conf=0.1)
