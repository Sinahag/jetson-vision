import cv2
import numpy as np

# Load pre-trained object detection model (e.g., YOLO)
model_config = "path/to/model/config.cfg"
model_weights = "path/to/model/weights.weights"
net = cv2.dnn.readNet(model_weights, model_config)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize variables for object tracking
tracker_list = []

# Initialize camera capture using GStreamer
camera = cv2.VideoCapture("nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Camera not opened.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize variables for detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Process detection outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Person class
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform object tracking
    for box in boxes:
        x, y, w, h = box
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x, y, w, h))
        tracker_list.append(tracker)

    # Update trackers and draw bounding boxes
    for tracker in tracker_list:
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(i) for i in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
camera.release()
cv2.destroyAllWindows()