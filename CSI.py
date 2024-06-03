import numpy as np
import cv2
import serial

window_title = "Person Detect"

serial_port = '/dev/ttyTHS1'
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate, timeout=1)

def gstreamer_pipeline(sensor_id=0, capture_width=640, capture_height=480, framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
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
            capture_width,
            capture_height,
        )
    )

pipeline0 = gstreamer_pipeline(sensor_id=0)
pipeline1 = gstreamer_pipeline(sensor_id=1)

def person_detect():
    body_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_upperbody.xml"
    )

    video_capture0 = cv2.VideoCapture(pipeline0, cv2.CAP_GSTREAMER)
    video_capture1 = cv2.VideoCapture(pipeline1, cv2.CAP_GSTREAMER)
    frame_width = 640

    if video_capture0.isOpened() and video_capture1.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret0, frame0 = video_capture0.read()
                ret1, frame1 = video_capture1.read()

                if not ret0 or not ret1:
                    print("Failed to capture image")
                    break

                gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                faces0 = body_cascade.detectMultiScale(gray0, 1.3, 5)
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                faces1 = body_cascade.detectMultiScale(gray1, 1.3, 5)
                
                pairs = []
                
                for (x, y, w, h) in faces0:
                    pairs.append([x, w])
                    cv2.rectangle(frame0, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                for (x, y, w, h) in faces1:
                    pairs.append([x, w])
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if len(pairs) > 0 and len(pairs) % 2 == 0:
                    x_diff = pairs[1][0] - pairs[0][0]
                    w_diff = pairs[1][1] - pairs[0][1]
                    x_loc = x_diff / 2 + pairs[1][0]
                    depth = int((270 / x_diff) * 10)  # in 1/10 centimeters
                    x_mean = x_diff / 2 + pairs[1][0]
                    x_offset = x_mean - frame_width / 2
                    angle = int(x_offset / 8)
                    print(f"Angle: {angle}")
                    print(f"Distance: {depth * 10}")
                    # send the angle with a 30 pt increase (avoid sending negatives)
                    scaled_angle = angle + 30
                    packet = depth.to_bytes(1, byteorder="big") + scaled_angle.to_bytes(1, byteorder="big")
                    ser.write(packet)

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title + "_right", frame0)
                    cv2.imshow(window_title + "_left", frame1)
                else:
                    break
                
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            ser.close()
            video_capture0.release()
            video_capture1.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    person_detect()
