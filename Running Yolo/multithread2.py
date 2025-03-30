from flask import Flask, jsonify
from collections import defaultdict
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)


class VideoProcessor:
    def __init__(self, video_path, model_path="../Yolo-Weights/best(1).pt"):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']  # Normal vehicles
        self.emergency_classes = ['ambulance']  # Emergency vehicles

    def process_video(self):
        results_dict = {}  # Format: {"0:30": {"normal": X, "emergency": Y}, ...}

        try:
            if not os.path.exists(self.video_path):
                print(f"Error: File not found - {self.video_path}")
                return results_dict

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video - {self.video_path}")
                return results_dict

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default FPS if not available

            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / fps if fps > 0 else frame_count / 30

                # Only process frames at 30-second intervals
                if current_time % 30 < 1 / (fps if fps > 0 else 30):
                    results = self.model(frame)
                    normal_count = 0
                    emergency_count = 0

                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            class_id = int(box.cls)
                            class_name = self.model.names[class_id]

                            if class_name.lower() in [v.lower() for v in self.vehicle_classes]:
                                normal_count += 1
                            elif class_name.lower() in [v.lower() for v in self.emergency_classes]:
                                emergency_count += 1

                    timestamp = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
                    results_dict[timestamp] = {
                        "normal": normal_count,
                        "emergency": emergency_count
                    }

                    print(f"\n{self.video_path} - {timestamp}:")
                    print(f"  Normal vehicles: {normal_count}")
                    print(f"  Emergency vehicles: {emergency_count}")

            cap.release()
            return results_dict

        except Exception as e:
            print(f"Error processing {self.video_path}: {str(e)}")
            return results_dict


@app.route('/get_results', methods=['GET'])
def get_results():
    # List of video paths to process
    video_paths = [
        "./Videos/22.mp4",
        "./Videos/amb1.mp4",
        "./Videos/vehicles.mp4",
        "./Videos/amb.mp4",
    ]

    # Verify paths exist before processing
    valid_paths = [path for path in video_paths if os.path.exists(path)]
    if len(valid_paths) != len(video_paths):
        missing = set(video_paths) - set(valid_paths)
        print(f"Warning: The following files were not found and will be skipped: {missing}")

    # Process each video and collect results
    results = {}
    for path in valid_paths:
        processor = VideoProcessor(path)
        video_results = processor.process_video()
        results[path] = video_results

    # Prepare the final response
    response = {
        'status': 'complete',
        'results': results
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)














# import cv2
# from ultralytics import YOLO
# import threading
# from collections import defaultdict
# import os
#
#
# class VideoProcessor:
#     def __init__(self, video_path, model_path="../Yolo-Weights/best(1).pt"):
#         self.video_path = video_path
#         self.model = YOLO(model_path)
#         # Define the vehicle class names your model recognizes
#         self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'ambulance']  # Add all your vehicle classes
#         self.counts = defaultdict(int)  # {timestamp: count}
#         self.detailed_counts = defaultdict(lambda: defaultdict(int))  # {timestamp: {class_name: count}}
#         self.running = True
#
#     def process_video(self):
#         try:
#             # Check if file exists
#             if not os.path.exists(self.video_path):
#                 print(f"Error: File not found - {self.video_path}")
#                 return {}, {}
#
#             cap = cv2.VideoCapture(self.video_path)
#             if not cap.isOpened():
#                 print(f"Error: Could not open video - {self.video_path}")
#                 return {}, {}
#
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             if fps <= 0:
#                 fps = 30  # Default FPS if not available
#                 print(f"Warning: Invalid FPS for {self.video_path}, using default {fps} FPS")
#
#             frame_count = 0
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             duration = total_frames / fps if fps > 0 else 0
#
#             print(f"Processing {self.video_path} (FPS: {fps:.2f}, Duration: {duration:.2f}s)")
#
#             while cap.isOpened() and self.running:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#
#                 frame_count += 1
#                 current_time = frame_count / fps if fps > 0 else frame_count / 30  # Fallback to 30 FPS
#
#                 # Only process frames at 30-second intervals
#                 if current_time % 30 < 1 / (fps if fps > 0 else 30):
#                     results = self.model(frame)
#                     vehicle_count = 0
#
#                     for result in results:
#                         boxes = result.boxes
#                         for box in boxes:
#                             class_id = int(box.cls)
#                             class_name = self.model.names[class_id]
#
#                             if class_name.lower() in [v.lower() for v in self.vehicle_classes]:
#                                 vehicle_count += 1
#                                 timestamp = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
#                                 self.detailed_counts[timestamp][class_name] += 1
#
#                     timestamp = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
#                     self.counts[timestamp] = vehicle_count
#
#                     # Print detailed information for this timestamp
#                     print(f"\n{self.video_path} - {timestamp}: {vehicle_count} vehicles")
#                     for class_name, count in self.detailed_counts[timestamp].items():
#                         print(f"  {class_name}: {count}")
#
#             cap.release()
#             return self.counts, self.detailed_counts
#
#         except Exception as e:
#             print(f"Error processing {self.video_path}: {str(e)}")
#             return {}, {}
#
#
# def process_video_thread(video_path, results_dict, detailed_results_dict):
#     processor = VideoProcessor(video_path)
#     counts, detailed_counts = processor.process_video()
#     results_dict[video_path] = counts
#     detailed_results_dict[video_path] = detailed_counts
#
#
# if __name__ == "__main__":
#     # List of video paths to process
#     video_paths = [
#         "./Videos/22.mp4",
#         "./Videos/amb1.mp4",
#         "./Videos/vehicles.mp4",
#         "./Videos/amb.mp4",
#     ]
#
#     # Verify paths exist before processing
#     valid_paths = [path for path in video_paths if os.path.exists(path)]
#     if len(valid_paths) != len(video_paths):
#         missing = set(video_paths) - set(valid_paths)
#         print(f"Warning: The following files were not found and will be skipped: {missing}")
#
#     # Dictionaries to store results from all threads
#     results = {}
#     detailed_results = {}
#     threads = []
#
#     # Start processing each video in a separate thread
#     for path in valid_paths:
#         thread = threading.Thread(
#             target=process_video_thread,
#             args=(path, results, detailed_results)
#         )
#         threads.append(thread)
#         thread.start()
#
#     # Wait for all threads to complete
#     for thread in threads:
#         thread.join()
#
#     # Print final summary
#     print("\nFinal Vehicle Count Summary:")
#     for video_path, counts in results.items():
#         if not counts:  # Skip videos that had errors
#             continue
#         print(f"\n{video_path}:")
#         for timestamp in sorted(counts.keys()):
#             print(f"\n  {timestamp}: {counts[timestamp]} vehicles")
#             for class_name, count in detailed_results[video_path][timestamp].items():
#                 print(f"    {class_name}: {count}")