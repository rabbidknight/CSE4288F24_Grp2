from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize model
    model = YOLO('yolov8s.pt')

    # Train model
    model.train(data='yolo_dataset/data.yaml', epochs=50, imgsz=1024, batch=16)
