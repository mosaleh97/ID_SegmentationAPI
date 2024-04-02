from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # Load YOLOv8s model

model.train(data='config.yaml', epochs=100, imgsz=640, workers=0)
