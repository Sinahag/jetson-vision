import cv2
import numpy as np
from CSI import CSI_Camera


""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080
"""

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
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

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

class STREAM:
    def __init__(self):
        self.confidence_threshold=0.5
        self.NMS_threshold=0.5
        self.input_size=416
        ### uncomment the following if you want to be able to detect things other than people
        # self.object_names=0
        # with open("coco.names","r") as f:
        #     self.object_names=[cname.strip() for cname in f.readlines()]
        # print(f"objects:\n{self.object_names}")
        
        # Load pre-trained YOLO model
        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

    def run_cameras(self):
        window_title = "CSI Cameras"
        left_camera = CSI_Camera()
        left_camera.open(
            gstreamer_pipeline(
                sensor_id=0,
                capture_width=1920,
                capture_height=1080,
                flip_method=0,
                display_width=960,
                display_height=540,
            )
        )
        left_camera.start()

        right_camera = CSI_Camera()
        right_camera.open(
            gstreamer_pipeline(
                sensor_id=1,
                capture_width=1920,
                capture_height=1080,
                flip_method=0,
                display_width=960,
                display_height=540,
            )
        )
        right_camera.start()

        if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():
            # Open file for writing locations
            output_file = open("person_locations.txt", "w")

            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

            try:
                while True:
                    _, left_image = left_camera.read()
                    _, right_image = right_camera.read()

                    # Create a blob from the frame and perform a forward pass through the network
                    blob = cv2.dnn.blobFromImage(left_image, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
                    self.net.setInput(blob)
                    outs_l = self.net.forward(get_output_layers(self.net))

                    # Create a blob from the frame and perform a forward pass through the network
                    blob = cv2.dnn.blobFromImage(right_image, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
                    self.net.setInput(blob)
                    outs_r = self.net.forward(get_output_layers(self.net))

                    # Initialize lists for detected objects
                    boxes = []
                    confidences = []
                    class_ids = []

                    for frame in [left_image,right_image]:
                        outs = outs_r
                        if (frame == left_image).all():
                            outs=outs_l
                        for out in outs:
                            for detection in out:
                                scores = detection[5:]
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]
                                if confidence > self.confidence_threshold and class_id == 0:  # Person class
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

                    # Use numpy to place images next to each other
                    camera_images = np.hstack((left_image, right_image)) 
                    # Check to see if the user closed the window
                    # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                    # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                        cv2.imshow(window_title, camera_images)
                    else:
                        break

                    # This also acts as
                    keyCode = cv2.waitKey(30) & 0xFF
                    # Stop the program on the ESC key
                    if keyCode == 27:
                        break
            finally:

                left_camera.stop()
                left_camera.release()
                right_camera.stop()
                right_camera.release()
            cv2.destroyAllWindows()
        else:
            print("Error: Unable to open both cameras")
            left_camera.stop()
            left_camera.release()
            right_camera.stop()
            right_camera.release()



if __name__ == "__main__":
    s = STREAM()
    s.run_cameras()
