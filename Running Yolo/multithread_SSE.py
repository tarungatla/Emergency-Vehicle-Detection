from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import threading
from collections import defaultdict
import os
import time
import json

app = Flask(__name__)

class VideoProcessor:
    def __init__(self, video_path, model_path="../Yolo-Weights/best(1).pt"):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'ambulance']
        self.running = True

    def process_video(self):
        try:
            if not os.path.exists(self.video_path):
                print(f"Error: File not found - {self.video_path}")
                return

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video - {self.video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS is 0
            frame_count = 0

            while cap.isOpened() and self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / fps

                # Process at 30-second intervals
                if current_time % 30 < 1 / fps:
                    results = self.model(frame)
                    vehicle_count = 0
                    detailed_count = defaultdict(int)

                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            class_id = int(box.cls)
                            class_name = self.model.names[class_id]

                            if class_name.lower() in [v.lower() for v in self.vehicle_classes]:
                                vehicle_count += 1
                                detailed_count[class_name] += 1

                    timestamp = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
                    data = {
                        "video": os.path.basename(self.video_path),
                        "timestamp": timestamp,
                        "total_vehicles": vehicle_count,
                        "vehicles": dict(detailed_count)
                    }
                    # Send SSE event
                    yield f"data: {json.dumps(data)}\n\n"

            cap.release()

        except Exception as e:
            print(f"Error processing {self.video_path}: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<path:video_path>')
def video_feed(video_path):
    processor = VideoProcessor(video_path)
    return Response(processor.process_video(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)