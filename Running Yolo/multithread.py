import cv2
import cvzone
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
from typing import Optional, List, Tuple


class VideoAnalyzer(threading.Thread):
    """Thread class for analyzing individual video streams"""

    def __init__(
            self,
            video_path: str,
            model: YOLO,
            result_queue: queue.Queue,
            thread_id: int
    ):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.result_queue = result_queue
        self.thread_id = thread_id
        self.stopped = False

        # Class names for vehicle detection
        self.vehicle_classes = ["car", "truck", "bus", "motorbike", "bicycle", "ambulance"]
        self.classNames = [ "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "ambulance"
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           ]

        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.vehicle_count = 0
        self.tracked_vehicles = set()

        # Define counting line (middle of frame)
        self.frame_height = 0
        self.counting_line = 0

    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List, int]:
        """Run object detection on a single frame and count vehicles"""
        results = self.model(frame, stream=True)

        # Create copy of original frame for annotations
        annotated_frame = frame.copy()

        # Initialize vehicle count for this frame
        current_frame_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = round(box.conf[0].item(), 2)

                # Class Name
                cls = int(box.cls[0])
                class_name = self.classNames[cls]

                # Only process vehicles
                if class_name in self.vehicle_classes:
                    # Draw corner rectangle
                    cvzone.cornerRect(annotated_frame, (x1, y1, w, h))

                    # Calculate center point
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Count vehicle if crossing counting line
                    if cy > self.counting_line and (cx, cy) not in self.tracked_vehicles:
                        self.tracked_vehicles.add((cx, cy))
                        self.vehicle_count += 1
                        current_frame_count += 1

                    # Display label
                    cvzone.putTextRect(annotated_frame, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1)

        return annotated_frame, results, current_frame_count

    def run(self):
        """Main loop for video processing"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Error {self.thread_id}] Could not open video stream")
            return

        # Get frame height for counting line
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.counting_line = self.frame_height // 2

        while not self.stopped:
            self.new_frame_time = time.time()

            success, frame = cap.read()
            if not success:
                break

            # Perform object detection and counting
            annotated_frame, results, current_count = self.detect_objects(frame)

            # Calculate FPS
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time

            # Add FPS and vehicle count text
            cvzone.putTextRect(annotated_frame, f'FPS: {round(fps, 2)}',
                               (10, 50), scale=1, thickness=1)
            cvzone.putTextRect(annotated_frame, f'Vehicles: {self.vehicle_count}',
                               (10, 90), scale=1, thickness=1)

            # Draw counting line
            cv2.line(annotated_frame, (0, self.counting_line),
                     (annotated_frame.shape[1], self.counting_line),
                     (0, 255, 0), 2)

            # Put results in queue
            self.result_queue.put({
                'thread_id': self.thread_id,
                'frame': annotated_frame,
                'fps': round(fps, 2),
                'vehicle_count': self.vehicle_count,
                'results': results
            })

            # Limit processing speed to prevent buffer overflow
            time.sleep(0.033)  # ~30 FPS

        cap.release()

    def stop(self):
        """Stop the thread"""
        self.stopped = True


class VideoProcessingSystem:
    """Main system for managing multiple video analyzers"""

    def __init__(self, model: YOLO, video_paths: List[str]):
        self.model = model
        self.video_paths = video_paths
        self.result_queue = queue.Queue(maxsize=100)
        self.analyzers = []

    def start(self):
        """Start all video analyzers"""
        for idx, path in enumerate(self.video_paths[:4]):  # Limit to 4 videos
            analyzer = VideoAnalyzer(path, self.model, self.result_queue, idx)
            analyzer.start()
            self.analyzers.append(analyzer)

    def stop(self):
        """Stop all video analyzers"""
        for analyzer in self.analyzers:
            analyzer.stop()
        for analyzer in self.analyzers:
            analyzer.join()

    def process_results(self):
        """Process results from all analyzers"""
        windows = [f'Video {i}' for i in range(len(self.analyzers))]

        while True:
            try:
                result = self.result_queue.get(timeout=1)
                window_name = windows[result['thread_id']]

                # Display frame
                cv2.imshow(window_name, result['frame'])

                # Print FPS and vehicle count (optional)
                print(f"{window_name}: FPS={result['fps']}, Vehicles={result['vehicle_count']}")

            except queue.Empty:
                continue

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


def main(model: YOLO, video_paths: List[str]):
    """Main function to run the video analysis system"""
    processing_system = VideoProcessingSystem(model, video_paths)

    try:
        processing_system.start()
        processing_system.process_results()
    finally:
        processing_system.stop()


# Example usage
if __name__ == "__main__":
    # Initialize YOLOv8 model
    model = YOLO("../Yolo-Weights/best(1).pt")

    # List of video paths
    video_paths = [
        "./Videos/11.mp4",
        "./Videos/2.mp4",
        "./Videos/3.mp4",
        "./Videos/4.mp4"
    ]

    main(model, video_paths)









# import cv2
# import cvzone
# import threading
# import queue
# import time
# import numpy as np
# from ultralytics import YOLO
# from typing import Optional, List, Tuple
#
#
# class VideoAnalyzer(threading.Thread):
#     """Thread class for analyzing individual video streams"""
#
#     def __init__(
#             self,
#             video_path: str,
#             model: YOLO,
#             result_queue: queue.Queue,
#             thread_id: int
#     ):
#         super().__init__()
#         self.video_path = video_path
#         self.model = model
#         self.result_queue = result_queue
#         self.thread_id = thread_id
#         self.stopped = False
#
#         # Class names from your original code
#         self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#                            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#                            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
#                            "umbrella",
#                            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
#                            "baseball bat",
#                            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#                            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#                            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#                            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#                            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#                            "teddy bear", "hair drier", "toothbrush"]
#
#         self.prev_frame_time = 0
#         self.new_frame_time = 0
#
#     def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
#         """Run object detection on a single frame"""
#         results = self.model(frame, stream=True)
#
#         # Create copy of original frame for annotations
#         annotated_frame = frame.copy()
#
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 w, h = x2 - x1, y2 - y1
#
#                 # Draw corner rectangle
#                 cvzone.cornerRect(annotated_frame, (x1, y1, w, h))
#
#                 # Confidence
#                 conf = round(box.conf[0].item(), 2)
#
#                 # Class Name
#                 cls = int(box.cls[0])
#
#                 # Display label
#                 cvzone.putTextRect(annotated_frame, f'{self.classNames[cls]} {conf}',
#                                    (max(0, x1), max(35, y1)), scale=1, thickness=1)
#
#         return annotated_frame, results
#
#     def run(self):
#         """Main loop for video processing"""
#         cap = cv2.VideoCapture(self.video_path)
#         if not cap.isOpened():
#             print(f"[Error {self.thread_id}] Could not open video stream")
#             return
#
#         while not self.stopped:
#             self.new_frame_time = time.time()
#
#             success, frame = cap.read()
#             if not success:
#                 break
#
#             # Perform object detection
#             annotated_frame, results = self.detect_objects(frame)
#
#             # Calculate FPS
#             fps = 1 / (self.new_frame_time - self.prev_frame_time)
#             self.prev_frame_time = self.new_frame_time
#
#             # Add FPS text
#             cvzone.putTextRect(annotated_frame, f'FPS: {round(fps, 2)}',
#                                (10, 50), scale=1, thickness=1)
#
#             # Put results in queue
#             self.result_queue.put({
#                 'thread_id': self.thread_id,
#                 'frame': annotated_frame,
#                 'fps': round(fps, 2),
#                 'results': results
#             })
#
#             # Limit processing speed to prevent buffer overflow
#             time.sleep(0.033)  # ~30 FPS
#
#         cap.release()
#
#     def stop(self):
#         """Stop the thread"""
#         self.stopped = True
#
#
# class VideoProcessingSystem:
#     """Main system for managing multiple video analyzers"""
#
#     def __init__(self, model: YOLO, video_paths: List[str]):
#         self.model = model
#         self.video_paths = video_paths
#         self.result_queue = queue.Queue(maxsize=100)
#         self.analyzers = []
#
#     def start(self):
#         """Start all video analyzers"""
#         for idx, path in enumerate(self.video_paths[:4]):  # Limit to 4 videos
#             analyzer = VideoAnalyzer(path, self.model, self.result_queue, idx)
#             analyzer.start()
#             self.analyzers.append(analyzer)
#
#     def stop(self):
#         """Stop all video analyzers"""
#         for analyzer in self.analyzers:
#             analyzer.stop()
#         for analyzer in self.analyzers:
#             analyzer.join()
#
#     def process_results(self):
#         """Process results from all analyzers"""
#         windows = [f'Video {i}' for i in range(len(self.analyzers))]
#
#         while True:
#             try:
#                 result = self.result_queue.get(timeout=1)
#                 window_name = windows[result['thread_id']]
#
#                 # Display frame
#                 cv2.imshow(window_name, result['frame'])
#
#                 # Print FPS and detections (optional)
#                 print(f"{window_name}: FPS={result['fps']}")
#
#             except queue.Empty:
#                 continue
#
#             # Exit condition
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         cv2.destroyAllWindows()
#
#
# def main(model: YOLO, video_paths: List[str]):
#     """Main function to run the video analysis system"""
#     processing_system = VideoProcessingSystem(model, video_paths)
#
#     try:
#         processing_system.start()
#         processing_system.process_results()
#     finally:
#         processing_system.stop()
#
#
# # Example usage
# if __name__ == "__main__":
#     # Initialize YOLOv8 model
#     model = YOLO("../Yolo-Weights/best(1).pt")
#
#     # List of video paths
#     video_paths = [
#         "./Videos/1.mp4",
#         "./Videos/2.mp4",
#         "./Videos/3.mp4",
#         "./Videos/4.mp4"
#     ]
#
#     main(model, video_paths)