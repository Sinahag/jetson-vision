import cv2
import numpy as np

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set input image size and confidence threshold
input_size = 416
confidence_threshold = 0.5

# Initialize camera capture using GStreamer
camera = cv2.VideoCapture("nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Camera not opened.")
    exit()

# Open file for writing locations
output_file = open("person_locations.txt", "w")

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Create a blob from the frame and perform a forward pass through the network
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Process detection outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and class_id == 0:  # Person class
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

    # Write person locations to file
    for box in boxes:
        x, y, w, h = box
        output_file.write(f"Person location: ({x}, {y}), ({x+w}, {y+h})\n")

    # Draw bounding boxes around detected persons
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the output file
camera.release()
output_file.close()
cv2.destroyAllWindows()