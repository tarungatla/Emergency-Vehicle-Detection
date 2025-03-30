#Deepseek
import cv2
import cvzone
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
from typing import List, Dict


class VideoAnalyzer(threading.Thread):
    """Thread class for analyzing individual video streams with proper vehicle counting"""

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

        # Vehicle classes to detect (update according to your model)
        self.vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle", "ambulance"]
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light","ambulance"  # Add ambulance if not present
        ]

        # Tracking parameters
        self.tracked_vehicles: Dict[int, Dict] = {}
        self.next_vehicle_id = 1
        self.max_age = 10  # Frames to keep track without updates
        self.vehicle_count = 0

        # Counting line parameters
        self.frame_size = (640, 480)  # Default size, updated in run()
        self.counting_line_pos = 0.6  # 60% of frame height

    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        # Calculate intersection area
        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

        # Calculate union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with detection and tracking"""
        annotated_frame = frame.copy()

        # Resize frame for consistent processing
        frame = cv2.resize(frame, self.frame_size)

        # Run YOLO detection
        results = self.model(frame, verbose=False)

        current_detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                if conf < 0.5:  # Confidence threshold
                    continue

                cls_id = int(box.cls)
                class_name = self.class_names[cls_id]
                if class_name not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_detections.append({
                    'box': (x1, y1, x2, y2),
                    'class': class_name,
                    'conf': conf
                })

        # Update existing tracks
        active_ids = []
        for vehicle_id in list(self.tracked_vehicles.keys()):
            track = self.tracked_vehicles[vehicle_id]
            track['age'] += 1

            # Find best matching detection
            best_iou = 0.4
            best_match = None
            for idx, det in enumerate(current_detections):
                iou_score = self.iou(track['box'], det['box'])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_match = idx

            if best_match is not None:
                # Update track with detection
                det = current_detections.pop(best_match)
                track.update({
                    'box': det['box'],
                    'age': 0,
                    'class': det['class'],
                    'conf': det['conf']
                })
                active_ids.append(vehicle_id)

        # Add new detections as tracks
        for det in current_detections:
            self.tracked_vehicles[self.next_vehicle_id] = {
                'box': det['box'],
                'age': 0,
                'counted': False,
                'class': det['class'],
                'conf': det['conf']
            }
            self.next_vehicle_id += 1

        # Remove old tracks
        self.tracked_vehicles = {
            k: v for k, v in self.tracked_vehicles.items()
            if v['age'] < self.max_age
        }

        # Check counting line crossings
        counting_line_y = int(self.frame_size[1] * self.counting_line_pos)
        for vehicle_id, track in self.tracked_vehicles.items():
            x1, y1, x2, y2 = track['box']
            center_y = (y1 + y2) // 2

            # Draw bounding box
            cvzone.cornerRect(annotated_frame, (x1, y1, x2 - x1, y2 - y1))
            cvzone.putTextRect(
                annotated_frame,
                f"{track['class']} {track['conf']:.2f}",
                (max(0, x1), max(35, y1)),
                scale=0.8, thickness=1
            )

            # Check counting condition
            if not track['counted'] and center_y > counting_line_y:
                self.vehicle_count += 1
                track['counted'] = True

        # Draw counting line
        cv2.line(
            annotated_frame,
            (0, counting_line_y),
            (self.frame_size[0], counting_line_y),
            (0, 255, 0), 2
        )

        # Add counters
        cvzone.putTextRect(
            annotated_frame,
            f"Vehicles: {self.vehicle_count}",
            (10, 50),
            scale=1, thickness=1
        )

        return annotated_frame

    def run(self):
        """Main processing loop"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Error {self.thread_id}] Could not open video stream")
            return

        # Get video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (640, 480)  # Force resize for processing

        while not self.stopped:
            success, frame = cap.read()
            if not success:
                break

            start_time = time.time()
            processed_frame = self.process_frame(frame)
            fps = 1 / (time.time() - start_time + 1e-9)

            # Put results in queue
            self.result_queue.put({
                'thread_id': self.thread_id,
                'frame': processed_frame,
                'fps': fps,
                'vehicle_count': self.vehicle_count
            })

            # Maintain processing speed
            time.sleep(0.001)

        cap.release()

    def stop(self):
        """Stop the thread"""
        self.stopped = True


class VideoProcessingSystem:
    """Main system for managing multiple video analyzers"""

    def __init__(self, model: YOLO, video_paths: List[str]):
        self.model = model
        self.video_paths = video_paths
        self.result_queue = queue.Queue(maxsize=20)
        self.analyzers = []

    def start(self):
        """Start all video analyzers"""
        for idx, path in enumerate(self.video_paths[:4]):
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
        windows = [f'Lane {i + 1}' for i in range(len(self.analyzers))]

        try:
            while True:
                # Process all frames in queue
                while not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    window_name = windows[result['thread_id']]

                    # Resize frame for display
                    display_frame = cv2.resize(result['frame'], (640, 480))

                    # Add FPS overlay
                    cv2.putText(
                        display_frame,
                        f"FPS: {result['fps']:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.imshow(window_name, display_frame)

                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()


def main():
    """Main function to run the video analysis system"""
    # Initialize YOLO model
    model = YOLO("yolov8n.pt")  # Replace with your model path

    # List of video paths
    video_paths = [
        "./Videos/11.mp4",
        "./Videos/22.mp4",
        "./Videos/11.mp4",
        "./Videos/vehicles.mp4"
    ]

    processing_system = VideoProcessingSystem(model, video_paths)

    try:
        processing_system.start()
        processing_system.process_results()
    finally:
        processing_system.stop()


if __name__ == "__main__":
    main()






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
#         # Class names for vehicle detection
#         self.vehicle_classes = ["car", "truck", "bus", "motorbike", "bicycle", "ambulance"]
#         self.classNames = [ "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "ambulance"
#                            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#                            ]
#
#         self.prev_frame_time = 0
#         self.new_frame_time = 0
#         self.vehicle_count = 0
#         self.tracked_vehicles = set()
#
#         # Define counting line (middle of frame)
#         self.frame_height = 0
#         self.counting_line = 0
#
#     def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List, int]:
#         """Run object detection on a single frame and count vehicles"""
#         results = self.model(frame, stream=True)
#
#         # Create copy of original frame for annotations
#         annotated_frame = frame.copy()
#
#         # Initialize vehicle count for this frame
#         current_frame_count = 0
#
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 w, h = x2 - x1, y2 - y1
#
#                 # Confidence
#                 conf = round(box.conf[0].item(), 2)
#
#                 # Class Name
#                 cls = int(box.cls[0])
#                 class_name = self.classNames[cls]
#
#                 # Only process vehicles
#                 if class_name in self.vehicle_classes:
#                     # Draw corner rectangle
#                     cvzone.cornerRect(annotated_frame, (x1, y1, w, h))
#
#                     # Calculate center point
#                     cx = (x1 + x2) // 2
#                     cy = (y1 + y2) // 2
#
#                     # Count vehicle if crossing counting line
#                     if cy > self.counting_line and (cx, cy) not in self.tracked_vehicles:
#                         self.tracked_vehicles.add((cx, cy))
#                         self.vehicle_count += 1
#                         current_frame_count += 1
#
#                     # Display label
#                     cvzone.putTextRect(annotated_frame, f'{class_name} {conf}',
#                                        (max(0, x1), max(35, y1)), scale=1, thickness=1)
#
#         return annotated_frame, results, current_frame_count
#
#     def run(self):
#         """Main loop for video processing"""
#         cap = cv2.VideoCapture(self.video_path)
#         if not cap.isOpened():
#             print(f"[Error {self.thread_id}] Could not open video stream")
#             return
#
#         # Get frame height for counting line
#         self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.counting_line = self.frame_height // 2
#
#         while not self.stopped:
#             self.new_frame_time = time.time()
#
#             success, frame = cap.read()
#             if not success:
#                 break
#
#             # Perform object detection and counting
#             annotated_frame, results, current_count = self.detect_objects(frame)
#
#             # Calculate FPS
#             fps = 1 / (self.new_frame_time - self.prev_frame_time)
#             self.prev_frame_time = self.new_frame_time
#
#             # Add FPS and vehicle count text
#             cvzone.putTextRect(annotated_frame, f'FPS: {round(fps, 2)}',
#                                (10, 50), scale=1, thickness=1)
#             cvzone.putTextRect(annotated_frame, f'Vehicles: {self.vehicle_count}',
#                                (10, 90), scale=1, thickness=1)
#
#             # Draw counting line
#             cv2.line(annotated_frame, (0, self.counting_line),
#                      (annotated_frame.shape[1], self.counting_line),
#                      (0, 255, 0), 2)
#
#             # Put results in queue
#             self.result_queue.put({
#                 'thread_id': self.thread_id,
#                 'frame': annotated_frame,
#                 'fps': round(fps, 2),
#                 'vehicle_count': self.vehicle_count,
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
#                 # Print FPS and vehicle count (optional)
#                 print(f"{window_name}: FPS={result['fps']}, Vehicles={result['vehicle_count']}")
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
#         "./Videos/11.mp4",
#         "./Videos/2.mp4",
#         "./Videos/3.mp4",
#         "./Videos/4.mp4"
#     ]
#
#     main(model, video_paths)
#
#
#
#
