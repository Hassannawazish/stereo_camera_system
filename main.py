import cv2
import numpy as np
import glob
import json

# Define the checkerboard pattern size (number of inner corners)
CHECKERBOARD = (8, 10)  # Adjust this to your checkerboard
SQUARE_SIZE = 15  # Size of a square in mm (or any unit)

# Prepare object points (3D points in real-world space)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale by square size

# Arrays to store object points and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load images
images = glob.glob("calibration_images/*.jpg")  # Change the path to your images

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print calibration results
print("\nCamera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", distortion_coeffs)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)

# Save calibration parameters
calibration_data = {
    "camera_matrix": camera_matrix.tolist(),
    "distortion_coeffs": distortion_coeffs.tolist(),
    "rotation_vectors": [r.tolist() for r in rvecs],
    "translation_vectors": [t.tolist() for t in tvecs]
}

with open("camera_calibration.json", "w") as f:
    json.dump(calibration_data, f, indent=4)

print("\nCalibration complete. Parameters saved to 'camera_calibration.json'.")
