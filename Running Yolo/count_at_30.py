import cv2
from ultralytics import YOLO


def count_vehicles(video_path):
    # Load the YOLOv8 model
    model = YOLO("../Yolo-Weights/best(1).pt")

    # Define vehicle class IDs (COCO dataset: 2=car, 3=motorcycle, 5=bus, 7=truck)
    VEHICLE_CLASSES = {2, 3, 5, 7}

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
    print(f"Frames per second: {fps}")
    frame_interval = int(fps * 30)  # Frame interval for every 30 seconds

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            # Perform vehicle detection
            results = model(frame, verbose=False)
            # Count detected vehicles
            vehicle_count = sum(
                (int(cls) in VEHICLE_CLASSES) for r in results for cls in r.boxes.cls
            )

            # Calculate timestamp
            actual_time = frame_count / fps
            minutes = int(actual_time // 60)
            seconds = int(actual_time % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            print(f"Time: {timestamp} | Vehicles detected: {vehicle_count}")

        # Display the current frame
        cv2.imshow("Video Frame", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    video_path = "./Videos/vehicles.mp4"
    count_vehicles(video_path)








# import cv2
# from ultralytics import YOLO
#
#
# def count_vehicles(video_path):
#     # Load the YOLOv8 model
#     model = YOLO("../Yolo-Weights/best(1).pt")
#
#     # Define vehicle class IDs (COCO dataset: 2=car, 3=motorcycle, 5=bus, 7=truck)
#     VEHICLE_CLASSES = {2, 3, 5, 7}
#
#     # Open video file
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return
#
#     fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
#     print(fps)
#     frame_interval = int(fps * 30)  # Frame interval for every 30 seconds
#
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # End of video
#
#         if frame_count % frame_interval == 0:
#             # Perform vehicle detection
#             results = model(frame, verbose=False)
#             # Count detected vehicles
#             vehicle_count = sum(
#                 (int(cls) in VEHICLE_CLASSES) for r in results for cls in r.boxes.cls
#             )
#
#             # Calculate timestamp
#             actual_time = frame_count / fps
#             minutes = int(actual_time // 60)
#             seconds = int(actual_time % 60)
#             timestamp = f"{minutes:02d}:{seconds:02d}"
#
#             print(f"Time: {timestamp} | Vehicles detected: {vehicle_count}")
#
#         frame_count += 1
#
#     cap.release()
#
#
# if __name__ == "__main__":
#     video_path = "./Videos/22.mp4"
#     count_vehicles(video_path)
