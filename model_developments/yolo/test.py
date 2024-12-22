from ultralytics import YOLO

trained_model = YOLO('runs/detect/train4/weights/best.pt')
results = trained_model.predict(source="test3.jpg", save=True, imgsz=1024, conf=0.1)
