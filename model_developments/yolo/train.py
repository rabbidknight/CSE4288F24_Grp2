from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')
    model.train(data='yolo_dataset/data.yaml', epochs=50, imgsz=1024, batch=16)
