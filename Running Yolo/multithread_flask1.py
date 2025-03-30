# proper formatting of the response


from flask import Flask, jsonify
from collections import defaultdict
import os
import cv2
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class VideoProcessor:
    def __init__(self, video_path, model_path="../Yolo-Weights/best(1).pt"):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']  # Normal vehicles
        self.emergency_classes = ['ambulance']  # Emergency vehicles

    def process_video(self):
        results_dict = {}  # Will store {timestamp: {"normal": X, "emergency": Y}}

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

            frame_count = -1

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
        "./Videos/amb_merged.mp4",
        "./Videos/vehicles.mp4",
        "./Videos/amb1.mp4",
    ]

    # Verify paths exist before processing
    valid_paths = [path for path in video_paths if os.path.exists(path)]
    if len(valid_paths) != len(video_paths):
        missing = set(video_paths) - set(valid_paths)
        print(f"Warning: The following files were not found and will be skipped: {missing}")

    # This will store our final results in the requested format:
    # {"0:30": {"video1": counts, "video2": counts}, "1:00": {...}, ...}
    final_results = defaultdict(dict)

    # Process each video and organize results by timestamp
    for path in valid_paths:
        processor = VideoProcessor(path)
        video_results = processor.process_video()

        # Reorganize the data by timestamp
        for timestamp, counts in video_results.items():
            final_results[timestamp][path] = counts

    # Convert defaultdict to regular dict for JSON serialization
    response = {
        'status': 'complete',
        'results': dict(final_results)
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)