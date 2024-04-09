import cv2
import numpy as np

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize cameras
pipeline0 = " ! ".join(["v4l2src device=/dev/video0",
                       "video/x-raw, width=320, height=240, framerate=30/1",
                       "videoconvert",
                       "video/x-raw, format=(string)BGR",
                       "appsink"
                       ])

pipeline1 = " ! ".join(["v4l2src device=/dev/video1",
                       "video/x-raw, width=320, height=240, framerate=30/1",
                       "videoconvert",
                       "video/x-raw, format=(string)BGR",
                       "appsink"
                       ])

#cam1 = cv2.VideoCapture(pipeline0, cv2.CAP_GSTREAMER)
#cam2 = cv2.VideoCapture(pipeline1, cv2.CAP_GSTREAMER)

cam1= cv2.VideoCapture(0)
cam2= cv2.VideoCapture(1)

while True:
    # Capture frame from camera 1
    ret1, frame1 = cam1.read()
    if not ret1:
        break
    
    # Capture frame from camera 2
    ret2, frame2 = cam2.read()
    if not ret2:
        break

    # Resize frames for faster processing
    #frame1 = cv2.resize(frame1, None, fx=0.4, fy=0.4)
    #frame2 = cv2.resize(frame2, None, fx=0.4, fy=0.4)
    
    print("bing")

    # Detect objects in frame1S
    height, width, channels = frame1.shape
    blob1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob1)
    outs1 = net.forward(output_layers)

    # Get object information from frame1
    class_ids1 = []
    confidences1 = []
    boxes1 = []
    for out in outs1:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids1.append(class_id)
                confidences1.append(float(confidence))
                boxes1.append([x, y, w, h])

    # Detect objects in frame2
    height, width, channels = frame2.shape
    blob2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob2)
    outs2 = net.forward(output_layers)

    # Get object information from frame2
    class_ids2 = []
    confidences2 = []
    boxes2 = []
    for out in outs2:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids2.append(class_id)
                confidences2.append(float(confidence))
                boxes2.append([x, y, w, h])

    # Match objects between frames based on class and proximity
    matches = []
    for i in range(len(boxes1)):
        for j in range(len(boxes2)):
            if class_ids1[i] == class_ids2[j]:
                # Calculate distance between centroids
                centroid1 = (boxes1[i][0] + boxes1[i][2] / 2, boxes1[i][1] + boxes1[i][3] / 2)
                centroid2 = (boxes2[j][0] + boxes2[j][2] / 2, boxes2[j][1] + boxes2[j][3] / 2)
                distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
                if distance < 50:  # Adjust this threshold as needed
                    matches.append((i, j))

    # Calculate and print the distance of each matched object from the cameras
    for match in matches:
        idx1, idx2 = match
        centroid1 = (boxes1[idx1][0] + boxes1[idx1][2] / 2, boxes1[idx1][1] + boxes1[idx1][3] / 2)
        centroid2 = (boxes2[idx2][0] + boxes2[idx2][2] / 2, boxes2[idx2][1] + boxes2[idx2][3] / 2)
        distance_px = abs(centroid1[0] - centroid2[0])
        print("Distance from cameras:", distance_px)

    # Display frames with detected objects
    cv2.imshow("Frame1", frame1)
    #cv2.imshow("Frame2", frame2)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam1.release()
cam2.release()
cv2.destroyAllWindows()
