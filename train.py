from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8s.pt')

# Train model
model.train(data='yolo_dataset/data.yaml', epochs=10, imgsz=768, batch=16)
