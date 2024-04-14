import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Load pre-trained YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    return net

# Convert YOLO model to TensorRT format
def convert_to_tensorrt(net):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = 1
        parser.register_input("input", (3, 416, 416))
        parser.register_output("output")
        parser.parse_buffer(net.getLayerBytes(), network)
        engine = builder.build_cuda_engine(network)
    return engine

# Initialize cameras
cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

# Load YOLO model and convert to TensorRT
net = load_yolo_model()
engine = convert_to_tensorrt(net)

# Create execution context for TensorRT engine
context = engine.create_execution_context()

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
    frame1_resized = cv2.resize(frame1, (416, 416))
    frame2_resized = cv2.resize(frame2, (416, 416))

    # Preprocess frames
    blob1 = cv2.dnn.blobFromImage(frame1_resized, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    blob2 = cv2.dnn.blobFromImage(frame2_resized, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Allocate device memory for input images
    cuda.memcpy_htod_async(input_memory1, blob1, stream)
    cuda.memcpy_htod_async(input_memory2, blob2, stream)

    # Run inference
    context.execute_async(bindings=[int(input_memory1), int(output_memory1)], stream_handle=stream.handle)
    context.execute_async(bindings=[int(input_memory2), int(output_memory2)], stream_handle=stream.handle)

    # Synchronize the stream
    stream.synchronize()

    # Post-process output
    # Implement your post-processing logic here

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

