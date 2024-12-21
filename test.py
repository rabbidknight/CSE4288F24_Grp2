from ultralytics import YOLO

# Load trained model
trained_model = YOLO(
    'C:\\Users\\durub\\PycharmProjects\\YOLO-crosswalk\\runs\\detect\\train\\weights\\best3.pt'
)

# Run inference with higher confidence and adjusted IOU
results = trained_model.predict(
    source="test3.jpg",
    save=True,
    imgsz=1024,
    conf=0.35,  # Increase confidence threshold
    iou=0.5,   # Increase IOU threshold
    max_det=10  # Limit maximum detections per image
)
