import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import cv2
import numpy as np

GObject.threads_init()
Gst.init(None)

# Define GStreamer pipeline for stereo vision
pipeline_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink name=left_sink sync=false "
    
    "nvarguscamerasrc sensor-id=1 ! "
    "video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink name=right_sink sync=false "
)

pipeline = Gst.parse_launch(pipeline_str)

# Get sink elements
left_sink = pipeline.get_by_name('left_sink')
right_sink = pipeline.get_by_name('right_sink')

# Define callback function to process frames
def process_frame(sink, data):
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps()
    array = np.ndarray(
        (caps.get_structure(0).get_value("height"),
         caps.get_structure(0).get_value("width"),
         3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=np.uint8)
    return array

# Set callbacks for left and right sinks
left_sink.set_property('emit-signals', True)
left_sink.connect('new-sample', process_frame, left_sink)
right_sink.set_property('emit-signals', True)
right_sink.connect('new-sample', process_frame, right_sink)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# Load YOLOv4 Tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Stereo vision depth estimation parameters
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Main loop
try:
    while True:
        # Capture frames from left and right cameras
        left_frame = process_frame(left_sink, None)
        right_frame = process_frame(right_sink, None)
        
        # Perform stereo vision depth estimation
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(left_gray, right_gray)
        
        # Normalize disparity map for visualization
        depth_map = (disparity / 16.0).astype(np.float32)
        depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Perform YOLOv4 Tiny object detection on the left frame
        blob = cv2.dnn.blobFromImage(left_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected, process further
                    center_x = int(detection[0] * 640)
                    center_y = int(detection[1] * 480)
                    w = int(detection[2] * 640)
                    h = int(detection[3] * 480)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Calculate depth (distance) of the object using depth map
                    depth = depth_map[y:y+h, x:x+w].mean()  # Use mean depth value of the bounding box region
                    # Perform mapping to 3D space using camera calibration parameters and stereo geometry
                    # Mapping code goes here
        
        # Display frames
        cv2.imshow('Left Camera', left_frame)
        cv2.imshow('Right Camera', right_frame)
        cv2.imshow('Depth Map', depth_map)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    # Stop pipeline
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()