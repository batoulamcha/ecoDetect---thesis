from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model = YOLO("Yolo-Weights/yolov8n.pt")
#mask = cv2.imread("Images/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

new_limits = [20, 20, 1260, 700]
totalCount = 0
myColor = (255, 0, 255)

while True:
    success, img = cap.read()
    #imgRegion = cv2.bitwise_and(img, mask)

    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=5)
            # Confidence
            conf = (math.ceil(box.conf[0]*100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Draw Red Rectangle for the counter
            cv2.rectangle(img, (new_limits[0], new_limits[1]), (new_limits[2], new_limits[3]), (0, 0, 255), 5)

            if conf > 0.3:
                if currentClass == 'person':
                    myColor = (128, 0, 128)
                elif currentClass == 'cell phone':
                    myColor = (0, 0, 255)
                elif currentClass == 'bottle':
                    myColor = (128, 128, 128)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                                   scale=1.6, thickness=1, offset=3,
                                   colorT=(255, 255, 255), colorR=myColor)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack([detections, currentArray])

    resultsTracker = tracker.update(detections)

    objects_inside_rectangle = set()
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        if new_limits[0] < x1 < new_limits[2] and new_limits[1] < y1 < new_limits[3]:
            objects_inside_rectangle.add(Id)
            # Draw a rectangle and ID for the counted object
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'{int(Id)}', (max(0, x2), max(35, y1)),
                               scale=2, thickness=4, offset=20)
        else:
            # Remove the object ID from the set if it exits the rectangle
            if Id in objects_inside_rectangle:
                objects_inside_rectangle.remove(Id)

    cvzone.putTextRect(img, f'Counter: {len(objects_inside_rectangle)}', (50, 50), scale=1, thickness=1, offset=20)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
