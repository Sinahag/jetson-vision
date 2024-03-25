import numpy as np
import cv2

# Load stereo calibration parameters
stereo_calibration_data = np.load('stereo_calibration.npz')
mtx1 = stereo_calibration_data['mtx1']
dist1 = stereo_calibration_data['dist1']
mtx2 = stereo_calibration_data['mtx2']
dist2 = stereo_calibration_data['dist2']
R = stereo_calibration_data['R']
T = stereo_calibration_data['T']

# Define the projection matrices (P) after rectification
# These are needed for computing the disparity map and the depth map
P1 = np.hstack((mtx1, np.zeros((3, 1))))
P2 = np.hstack((mtx2, T.reshape(3, 1)))

# Capture images from both cameras
cap1 = cv2.VideoCapture(0)  # Use the appropriate camera index for camera 1
cap2 = cv2.VideoCapture(1)  # Use the appropriate camera index for camera 2

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Undistort the images
    undistorted_frame1 = cv2.undistort(frame1, mtx1, dist1)
    undistorted_frame2 = cv2.undistort(frame2, mtx2, dist2)

    # Rectify the images
    map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R[0], P1, frame1.shape[::-1], cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R[1], P2, frame2.shape[::-1], cv2.CV_32FC1)
    rectified_frame1 = cv2.remap(undistorted_frame1, map1x, map1y, cv2.INTER_LINEAR)
    rectified_frame2 = cv2.remap(undistorted_frame2, map2x, map2y, cv2.INTER_LINEAR)

    # Compute disparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    gray1 = cv2.cvtColor(rectified_frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rectified_frame2, cv2.COLOR_BGR2GRAY)
    disparity_map = stereo.compute(gray1, gray2)

    # Compute depth map
    Q = stereo_calibration_data['Q']
    depth_map = cv2.reprojectImageTo3D(disparity_map, Q)

    # Normalize depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Stack rectified frames and depth map together horizontally
    output_frame = np.hstack((rectified_frame1, rectified_frame2, depth_map_normalized))

    # Display the result
    cv2.imshow('Stereo Vision', output_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras
cap1.release()
cap2.release()
cv2.destroyAllWindows()
