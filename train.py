from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data="yolo_dataset/data.yaml",
    epochs=50,
    lr0=0.001,  # Lower learning rate
    imgsz=1024,  # Larger image size
    batch=8
)
