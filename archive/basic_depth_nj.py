import cv2
import numpy as np

def list_ports():
    """
    Test the ports and returns a tuple with the available ports 
    and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports

print(list_ports())

# Load pre-trained YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize cameras
cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

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
    frame1 = cv2.resize(frame1, None, fx=0.4, fy=0.4)
    frame2 = cv2.resize(frame2, None, fx=0.4, fy=0.4)
    
    # Perform object detection on frame1
    blob1 = cv2.dnn.blobFromImage(frame1, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob1)
    outs1 = net.forward(output_layers)

    # Perform object detection on frame2
    blob2 = cv2.dnn.blobFromImage(frame2, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob2)
    outs2 = net.forward(output_layers)

    # Process detections for frame1
    class_ids1 = []
    confidences1 = []
    boxes1 = []
    for out in outs1:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame1.shape[1])
                center_y = int(detection[1] * frame1.shape[0])
                w = int(detection[2] * frame1.shape[1])
                h = int(detection[3] * frame1.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids1.append(class_id)
                confidences1.append(float(confidence))
                boxes1.append([x, y, w, h])

    # Process detections for frame2
    class_ids2 = []
    confidences2 = []
    boxes2 = []
    for out in outs2:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame2.shape[1])
                center_y = int(detection[1] * frame2.shape[0])
                w = int(detection[2] * frame2.shape[1])
                h = int(detection[3] * frame2.shape[0])
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
    cv2.imshow("Frame2", frame2)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam1.release()
cam2.release()
cv2.destroyAllWindows()
