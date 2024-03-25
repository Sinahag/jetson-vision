import numpy as np
import cv2

# Number of inner corners in the chessboard pattern
pattern_size = (9, 6)  # Change this to match your chessboard
gray1=0
# Termination criteria for corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
object_points = []  # 3D points in real world space
image_points1 = []  # 2D points in image plane for camera 1
image_points2 = []  # 2D points in image plane for camera 2

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Capture images from both cameras
cap1 = cv2.VideoCapture(0)  # Use the appropriate camera index for camera 1
cap2 = cv2.VideoCapture(1)  # Use the appropriate camera index for camera 2

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners for camera 1
    ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)

    # Find the chessboard corners for camera 2
    ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

    # If corners found in both cameras, add object points and image points
    if ret1 and ret2:
        object_points.append(objp)

        # Refine corner locations
        corners_refined1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        corners_refined2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

        image_points1.append(corners_refined1)
        image_points2.append(corners_refined2)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame1, pattern_size, corners_refined1, ret1)
        cv2.drawChessboardCorners(frame2, pattern_size, corners_refined2, ret2)
        cv2.imshow('Chessboard Calibration Camera 1', frame1)
        cv2.imshow('Chessboard Calibration Camera 2', frame2)

        # Wait for a key press to capture next image
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

# Release the cameras
cap1.release()
cap2.release()
cv2.destroyAllWindows()

# Perform stereo calibration
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(object_points, image_points1, image_points2, gray1.shape[::-1], None, None, None, criteria=criteria)

# Save the stereo calibration parameters
np.savez('stereo_calibration.npz', camera_matrix1=mtx1, distortion_coefficients1=dist1, camera_matrix2=mtx2, distortion_coefficients2=dist2, R=R, T=T, E=E, F=F)

print("Stereo calibration complete. Calibration parameters saved.")