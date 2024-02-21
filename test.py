from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

classNames = ["paper"]
#classNames = ["Watch"]
model = YOLO('runs/detect/newest/best.pt')
myColor = (255, 0, 255)

detected_objects = []

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    frame_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=5)
            # Confidence
            conf = (math.ceil(box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            frame_objects.append((currentClass, (x1, y1, x2, y2), conf))

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                               scale=1.6, thickness=1, offset=3,
                               colorT=(255, 255, 255), colorR=myColor)

    if frame_objects:
        detected_objects.extend(frame_objects)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

#model.predict(source="0", show=True, conf=0.5)
