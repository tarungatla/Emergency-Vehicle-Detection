from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model
model = YOLO("../Yolo-Weights/best(1).pt").to(device)

# Video capture
cap = cv2.VideoCapture("./Videos/11.mp4")

# Frame size
frame_width = 640
frame_height = 480

# Vehicle tracking dictionary
tracked_vehicles = {}
next_vehicle_id = 1  # Unique ID for vehicles
vehicle_count = 0


def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou_value = interArea / float(box1Area + box2Area - interArea)
    return iou_value


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    # Perform detection
    results = model(frame, device=device)
    new_tracked_vehicles = {}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            new_box = (x1, y1, x2, y2)
            vehicle_detected = False

            # Check if this vehicle is already being tracked
            for vehicle_id, old_box in tracked_vehicles.items():
                if iou(new_box, old_box) > 0.5:  # If similar, update
                    new_tracked_vehicles[vehicle_id] = new_box
                    vehicle_detected = True
                    break

            if not vehicle_detected:  # New vehicle detected
                new_tracked_vehicles[next_vehicle_id] = new_box
                vehicle_count += 1  # Increment only for first-time detection
                next_vehicle_id += 1

    tracked_vehicles = new_tracked_vehicles  # Update tracked vehicles

    # Draw bounding boxes
    for _, box in tracked_vehicles.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display total count
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()








# from ultralytics import YOLO
# import torch
# import cv2
# import numpy as np
#
# # Device selection
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
#
# # Load YOLO model
# model = YOLO("../Yolo-Weights/best(1).pt").to(device)
#
# # Video capture
# cap = cv2.VideoCapture("./Videos/22.mp4")
#
# # Frame size
# frame_width = 640
# frame_height = 480
#
# # Persistent vehicle tracking
# tracked_vehicles = {}
# total_count = 0
# frame_lifetime = 30  # Number of frames a vehicle is tracked
#
# def calculate_centroid(x1, y1, x2, y2):
#     return ((x1 + x2) // 2, (y1 + y2) // 2)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame = cv2.resize(frame, (frame_width, frame_height))
#
#     # Perform detection
#     results = model(frame, device=device)
#     current_centroids = []
#
#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             centroid = calculate_centroid(x1, y1, x2, y2)
#             current_centroids.append(centroid)
#
#             # Check if centroid is new
#             if not any(np.linalg.norm(np.array(centroid) - np.array(c)) < 50 for c in tracked_vehicles):
#                 tracked_vehicles[centroid] = frame_lifetime
#                 total_count += 1  # Increment only for new vehicles
#
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     # Reduce lifetime and remove old vehicles
#     for key in list(tracked_vehicles.keys()):
#         tracked_vehicles[key] -= 1
#         if tracked_vehicles[key] <= 0:
#             del tracked_vehicles[key]
#
#     # Display total count
#     cv2.putText(frame, f"Vehicles: {total_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#
#     cv2.imshow("Traffic Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# from ultralytics import YOLO
# import torch
# import cv2
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
#
# model = YOLO("../Yolo-Weights/best(1).pt").to(device)
#
# cap = cv2.VideoCapture("./Videos/22.mp4")
#
# frame_width = 640
# frame_height = 480
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame = cv2.resize(frame, (frame_width, frame_height))
#
#     results = model(frame, device=device)
#     vehicle_count = len(results[0].boxes)
#
#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#
#     cv2.imshow("Traffic Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()