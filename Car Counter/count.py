import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("../Yolo-Weights/best.pt")  # Use 'yolov8n.pt' or trained weights

# Open video file
video_path = "./Videos/1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# Define vehicle classes in COCO dataset
vehicle_classes = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

# Initialize counter
vehicle_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO inference
    results = model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id in vehicle_classes:
                vehicle_count += 1  # Increment count for each detected vehicle

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicle: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
