from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Set the webcam (try changing the index if needed)
cap = cv2.VideoCapture("./Videos/amb.mp4")  # For default webcam (0 or 1 depending on your setup)
# cap.set(3, 1280)
# cap.set(4, 720)

# Load YOLOv8 model
model = YOLO("../Yolo-Weights/best(1).pt")

# Class names
classNames = ["ambulance", "bicycle", "car", "motorbike", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat"

              ]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = round(box.conf[0].item(), 2)

            # Class Name
            cls = int(box.cls[0])

            # Display label
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Display FPS on frame
    cvzone.putTextRect(img, f'FPS: {round(fps, 2)}', (10, 50), scale=1, thickness=1)

    # Show Image
    cv2.imshow("Image", img)

    # Exit gracefully on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

