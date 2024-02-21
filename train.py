from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(data="configtest.yaml", epochs=50)
