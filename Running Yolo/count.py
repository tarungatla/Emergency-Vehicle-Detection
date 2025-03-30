import cv2
import numpy as np
from ultralytics import YOLO


def setup_vehicle_counter(video_path, confidence_threshold=0.5):
    """
    Initialize the vehicle counter with YOLO model and video capture.

    Args:
        video_path (str): Path to the input video
        confidence_threshold (float): Minimum confidence for detection

    Returns:
        tuple: (YOLO model, VideoCapture object)
    """
    # Initialize YOLO model
    model = YOLO("../Yolo-Weights/best(1).pt")

    # Load video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    return model, cap


def process_frame(frame, model, confidence_threshold=0.5):
    """
    Process a single frame to detect vehicles with improved tracking.

    Args:
        frame (numpy.ndarray): Input frame
        model (YOLO): YOLO model instance
        confidence_threshold (float): Minimum confidence for detection

    Returns:
        tuple: (List of vehicle detections, List of vehicle positions)
    """
    # Run YOLO detection
    results = model(frame)

    # Get vehicle detections
    vehicle_detections = []
    vehicle_positions = []

    for result in results:
        for detection in result.boxes:
            if detection.conf > confidence_threshold:
                x1 = int(detection.xyxy[0].item())
                y1 = int(detection.xyxy[1].item())
                x2 = int(detection.xyxy[2].item())
                y2 = int(detection.xyxy[3].item())

                cls = int(detection.cls.item())

                # Only consider vehicle classes
                if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    vehicle_detections.append((x1, y1, x2, y2))
                    vehicle_positions.append(((x1 + x2) // 2, (y1 + y2) // 2))

    return vehicle_detections, vehicle_positions


def count_vehicles(video_path):
    """
    Count vehicles moving from top to bottom in a video with improved tracking.

    Args:
        video_path (str): Path to input video

    Returns:
        int: Total count of vehicles
    """
    model, cap = setup_vehicle_counter(video_path)

    # Initialize tracking variables
    vehicle_count = 0
    tracked_positions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections, positions = process_frame(frame, model)

        # Draw detection line
        line_y = frame.shape[0] // 2
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

        # Draw detections
        for pos in positions:
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 4, (0, 255, 0), -1)

        # Update tracking
        for pos in positions:
            # Check if vehicle crossed center line
            if pos[1] > line_y and \
                    not any(abs(pos[0] - old_pos[0]) < 50 and
                            abs(pos[1] - old_pos[1]) < 50 for old_pos in tracked_positions):
                vehicle_count += 1

        tracked_positions.extend(positions)
        if len(tracked_positions) > 100:
            tracked_positions = tracked_positions[-50:]

        # Display frame count and vehicle count
        cv2.putText(frame, f'Frame: {frame_count}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Vehicle Counter', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return vehicle_count


# Example usage
video_path = "./Videos/amb1.mp4"
vehicle_count = count_vehicles(video_path)
print(f"Total vehicles counted: {vehicle_count}")