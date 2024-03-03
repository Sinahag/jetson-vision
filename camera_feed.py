import cv2

# Initialize camera capture using GStreamer
camera_0 = cv2.VideoCapture("nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
camera_1 = cv2.VideoCapture("nvarguscamerasrc sensor_id=1 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Check if cameras opened successfully
if not camera_0.isOpened() or not camera_1.isOpened():
    print("Error: Cameras not opened.")
    exit()

while True:
    # Capture frame-by-frame from camera 0
    ret_0, frame_0 = camera_0.read()

    # Capture frame-by-frame from camera 1
    ret_1, frame_1 = camera_1.read()

    # Check if frames were captured successfully
    if not ret_0 or not ret_1:
        print("Error: Frames not received.")
        break

    # Display frames from both cameras
    cv2.imshow('Camera 0', frame_0)
    cv2.imshow('Camera 1', frame_1)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the captures
camera_0.release()
camera_1.release()
cv2.destroyAllWindows()