from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/best(1).pt')
results = model("Images/amb2.jpeg", show=True)
cv2.waitKey(0)