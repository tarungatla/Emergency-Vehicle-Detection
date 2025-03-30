from flask import Flask, jsonify, request
import threading
from collections import defaultdict
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Global variables to store processing results and status
processing_results = {}
is_processing = False


class VideoProcessor:
    def __init__(self, video_path, model_path="../Yolo-Weights/best(1).pt"):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']  # Normal vehicles
        self.emergency_classes = ['ambulance']  # Emergency vehicles
        self.running = True

    def process_video(self):
        try:
            if not os.path.exists(self.video_path):
                print(f"Error: File not found - {self.video_path}")
                return {}

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video - {self.video_path}")
                return {}

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default FPS if not available

            frame_count = 0
            results_dict = {}  # Format: {"0:30": {"normal": X, "emergency": Y}, ...}

            while cap.isOpened() and self.running:
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
            return {}


def process_videos():
    global processing_results, is_processing

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

    # Reset results
    processing_results = {}
    is_processing = True

    # Process each video
    for path in valid_paths:
        processor = VideoProcessor(path)
        results = processor.process_video()
        processing_results[path] = results

    is_processing = False


@app.route('/start_processing', methods=['POST'])
def start_processing():
    global is_processing

    if is_processing:
        return jsonify({
            'status': 'error',
            'message': 'Processing is already in progress'
        }), 400

    # Start processing in a separate thread
    thread = threading.Thread(target=process_videos)
    thread.start()

    return jsonify({
        'status': 'success',
        'message': 'Video processing started'
    })


@app.route('/get_results', methods=['GET'])
def get_results():
    global processing_results, is_processing

    if is_processing:
        return jsonify({
            'status': 'processing',
            'message': 'Processing is still in progress'
        })

    if not processing_results:
        return jsonify({
            'status': 'ready',
            'message': 'No results available. Start processing first.'
        })

    # Prepare the response in the requested format
    response = {
        'status': 'complete',
        'results': processing_results
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)