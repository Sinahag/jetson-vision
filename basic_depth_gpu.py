import cv2
import numpy as np
import time

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize cameras
cam1 = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, format=BGRx ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
cam2 = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw, format=BGRx ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

while True:
    start_time = time.time()
    
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
    
    # Combine frames into a batch for parallel processing
    batch = np.stack((frame1, frame2), axis=0)
    
    # Detect objects in the batch
    blob = cv2.dnn.blobFromImages(batch, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Split detections for each frame
    num_detections = outs[0].shape[0]
    outs1 = outs[:, :num_detections // 2, ...]
    outs2 = outs[:, num_detections // 2:, ...]

    # Match objects between frames based on class and proximity
    matches = []
    for detection1 in outs1[0]:
        for detection2 in outs2[0]:
            scores1 = detection1[5:]
            class_id1 = np.argmax(scores1)
            confidence1 = scores1[class_id1]
            scores2 = detection2[5:]
            class_id2 = np.argmax(scores2)
            confidence2 = scores2[class_id2]
            if class_id1 == class_id2 and confidence1 > 0.5 and confidence2 > 0.5:
                # Calculate distance between centroids
                centroid1 = (detection1[0] * frame1.shape[1], detection1[1] * frame1.shape[0])
                centroid2 = (detection2[0] * frame2.shape[1], detection2[1] * frame2.shape[0])
                distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
                if distance < 50:  # Adjust this threshold as needed
                    matches.append((centroid1, centroid2))

    # Calculate and print the distance of each matched object from the cameras
    for match in matches:
        centroid1, centroid2 = match
        distance_px = abs(centroid1[0] - centroid2[0])
        print("Distance from cameras:", distance_px)

    # Display frames with detected objects
    cv2.imshow("Frame1", frame1)
    cv2.imshow("Frame2", frame2)
    
    # Delay to achieve 1 FPS
    elapsed_time = time.time() - start_time
    if elapsed_time < 1:
        time.sleep(1 - elapsed_time)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam1.release()
cam2.release()
cv2.destroyAllWindows()