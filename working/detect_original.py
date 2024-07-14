import cv2 as cv
import numpy as np
import serial
import struct

serial_port = '/dev/ttyTHS1'
baud_rate=9600
ser = serial.Serial(serial_port,baud_rate,timeout=1)


# Load names of classes
def get_output_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Draw bounding box
def draw_pred(class_id, conf, left, top, right, bottom, frame, classes):
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    label = '%.2f' % conf
    if classes:
        label = '%s:%s' % (classes[class_id], label)
    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Load class names
classes_file = "coco.names"
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load YOLOv4-tiny model
model_configuration = "yolov4-tiny.cfg"
model_weights = "yolov4-tiny.weights"
net = cv.dnn.readNetFromDarknet(model_configuration, model_weights)

# Check if CUDA backend is available
if cv.cuda.getCudaEnabledDeviceCount() > 0:
    print("number of available cuda cores: " + str(cv.cuda.getCudaEnabledDeviceCount()))
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
else:
    print("[WARN] CUDA backend not available, using CPU instead.")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Start G Streamer Pipelines
pipeline0 = gstreamer_pipeline(sensor_id=0)
pipeline1 = gstreamer_pipeline(sensor_id=1)

# Open video capture for two cameras
capR = cv.VideoCapture(pipeline0, cv.CAP_GSTREAMER)
capL = cv.VideoCapture(pipeline1, cv.CAP_GSTREAMER)

if not capL.isOpened() or not capR.isOpened():
    print("Error: Could not open cameras.")
    exit()

#cv.namedWindow("Object Detection Left", cv.WINDOW_AUTOSIZE)
#cv.namedWindow("Object Detection Right", cv.WINDOW_AUTOSIZE)

# Read stereo camera calibration parameters (placeholders, replace with actual parameters)
camera_matrixL = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
dist_coeffsL = np.zeros(5)
camera_matrixR = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
dist_coeffsR = np.zeros(5)
R = np.eye(3)
T = np.zeros((3, 1))
R1, R2, P1, P2, Q = [np.eye(3) for _ in range(5)]
frame_width = 1280

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Error: Could not read frame.")
        break

    blobL = cv.dnn.blobFromImage(frameL, 1/255.0, (416, 416), [0, 0, 0], 1, crop=False)
    blobR = cv.dnn.blobFromImage(frameR, 1/255.0, (416, 416), [0, 0, 0], 1, crop=False)
    net.setInput(blobL)
    outsL = net.forward(get_output_names(net))
    net.setInput(blobR)
    outsR = net.forward(get_output_names(net))

    conf_threshold = 0.5 # minimum confidence for detected object to be drawn around
    nms_threshold = 0.4
    frame_heightL, frame_widthL = frameL.shape[:2]
    frame_heightR, frame_widthR = frameR.shape[:2]

    class_idsL, confidencesL, boxesL, centersL = [], [], [], []
    class_idsR, confidencesR, boxesR, centersR= [], [], [], []

    # Process detections for left frame
    for out in outsL:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_widthL)
                center_y = int(detection[1] * frame_heightL)
                width = int(detection[2] * frame_widthL)
                height = int(detection[3] * frame_heightL)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_idsL.append(class_id)
                confidencesL.append(float(confidence))
                boxesL.append([left, top, width, height])

    # Apply non-maxima suppression for left frame
    indicesL = cv.dnn.NMSBoxes(boxesL, confidencesL, conf_threshold, nms_threshold)
    if len(indicesL) > 0:
        for i in indicesL.flatten():
            box = boxesL[i]
            left, top, width, height = box
            centersL.append([class_idsL[i],left+(width/2)])
            draw_pred(class_idsL[i], confidencesL[i], left, top, left + width, top + height, frameL, classes)

    # Process detections for right frame
    for out in outsR:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_widthR)
                center_y = int(detection[1] * frame_heightR)
                width = int(detection[2] * frame_widthR)
                height = int(detection[3] * frame_heightR)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_idsR.append(class_id)
                confidencesR.append(float(confidence))
                boxesR.append([left, top, width, height])

    # Apply non-maxima suppression for right frame
    indicesR = cv.dnn.NMSBoxes(boxesR, confidencesR, conf_threshold, nms_threshold)
    if len(indicesR) > 0:
        for i in indicesR:
            box = boxesR[i]
            left, top, width, height = box
            centersR.append([class_idsR[i],left+(width/2)])
            draw_pred(class_idsR[i], confidencesR[i], left, top, left + width, top + height, frameR, classes)
            
    if len(centersR) > 0 and len(centersR) == len(centersL): # if the same number of objects detected in both frames
        for i in range(len(centersL)):
            packet = bytes()
            if(centersL[i][0] == centersR[i][0]) and (centersL[i][0]==0): # checks if its the same object and if its a person
                x_diff = abs(centersL[i][1]-centersR[i][1])
                x_mean = centersR[i][1] + x_diff/2 - frame_width/2
                depth = int((270/x_diff)*100)
                angle = int(x_mean /16)
                if(angle<-30):
                    angle = -30
                scaled_angle = angle+30 # scaling this to a positive number to transmit to bbg and then will be scaled down on bbg side
                scaled_depth = int(depth/10) if (depth<2550) else int(255)
                print("person" + str(i) + " detected at: " +  str(depth) + "cm from launcher at: " + str(angle) + " degrees")
                packet+=scaled_depth.to_bytes(1,byteorder="big") +scaled_angle.to_bytes(1,byteorder="big")
            if packet:
                ser.write(packet)

    cv.imshow("Object Detection Left", frameL)
    cv.imshow("Object Detection Right", frameR)

    if cv.waitKey(1) >= 0:
        break

ser.close()
capL.release()
capR.release()
cv.destroyAllWindows()

