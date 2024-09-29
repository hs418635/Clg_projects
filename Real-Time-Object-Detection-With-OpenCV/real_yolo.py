# How to run:
# python real_time_object_detection.py --yolo yolo-coco --confidence 0.5

# Import packages
from imutils.video import VideoStream, FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak predictions")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Load class labels and YOLO model
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Check if files exist
if not os.path.isfile(labelsPath):
    raise FileNotFoundError(f"Labels file not found at {labelsPath}")
if not os.path.isfile(weightsPath):
    raise FileNotFoundError(f"Weights file not found at {weightsPath}")
if not os.path.isfile(configPath):
    raise FileNotFoundError(f"Config file not found at {configPath}")

print("[INFO] loading YOLO from disk...")
LABELS = open(labelsPath).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names = net.getLayerNames()

# Get YOLO output layers
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize class labels and colors
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

# Start video stream and FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Process frames
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    predictions = net.forward(output_layers)

    boxes = []
    confidences = []
    classIDs = []

    for output in predictions:
        for detection in output:
            detection = detection.reshape(-1, 85)  # Reshape to handle detection correctly
            for obj in detection:
                scores = obj[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > args["confidence"]:
                    box = obj[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()

# Stop FPS counter and clean up
fps.stop()
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
