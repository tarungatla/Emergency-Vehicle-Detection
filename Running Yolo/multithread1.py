import cv2
from ultralytics import YOLO
import threading
import time
from collections import defaultdict


class VideoProcessor:
    def __init__(self, video_path, model_path='yolov8n.pt'):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.counts = defaultdict(int)  # {timestamp: count}
        self.running = True

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing {self.video_path} (FPS: {fps:.2f}, Duration: {total_frames / fps:.2f}s)")

        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            # Only process frames at 30-second intervals
            if current_time % 30 < 1 / fps:  # Check if we're at a 30-second mark
                results = self.model(frame)
                vehicle_count = 0

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls)
                        if class_id in self.vehicle_classes:
                            vehicle_count += 1

                timestamp = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
                self.counts[timestamp] = vehicle_count

                print(f"{self.video_path} - {timestamp}: {vehicle_count} vehicles")

            # Early exit for demonstration
            if frame_count > total_frames:
                break

        cap.release()
        return self.counts


def process_video_thread(video_path, results_dict):
    processor = VideoProcessor(video_path)
    counts = processor.process_video()
    results_dict[video_path] = counts


if __name__ == "__main__":
    # List of video paths to process
    video_paths = [
        "./Videos/22.mp4",
        "./Videos/amb1.mp4",
        "./Videos/vehicles.mp4",
        "./Videos/22.mp4",
    ]

    # Replace with your actual video paths
    # video_paths = [f'traffic_{i}.mp4' for i in range(1, 5)]

    # Dictionary to store results from all threads
    results = {}
    threads = []

    # Start processing each video in a separate thread
    for path in video_paths:
        thread = threading.Thread(target=process_video_thread, args=(path, results))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Print final summary
    print("\nFinal Vehicle Count Summary:")
    for video_path, counts in results.items():
        print(f"\n{video_path}:")
        for timestamp, count in sorted(counts.items()):
            print(f"  {timestamp}: {count} vehicles")