import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
import threading
import os


def load_calibration_data(file_path: str) -> tuple:
    """
    Load stereo calibration data from a JSON file.
    
    Parameters:
        file_path (str): Path to the JSON file containing stereo camera calibration data.

    Returns:
        tuple: Contains camera matrices (K_left, K_right), distortion coefficients (D_left, D_right),
               image size, rotation (R), translation (T), rectification matrices (R1, R2),
               projection matrices (P1, P2), and the disparity-to-depth mapping matrix (Q).
    
    Raises:
        FileNotFoundError: If the file at `file_path` does not exist.
    """
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    with open(file_path, 'r') as f:
        calibration_data = json.load(f)
    
    # Camera matrix and distortion coefficients
    K_left = np.array(calibration_data['K_left'])
    D_left = np.array(calibration_data['D_left'])
    K_right = np.array(calibration_data['K_right'])
    D_right = np.array(calibration_data['D_right'])
    image_size = tuple(calibration_data['image_size'])

    # Extract the stereo calibration data
    R = np.eye(3)
    T = np.array([[-0.1], [0.0], [0.0]])
    R1 = np.array(calibration_data['R1'])
    R2 = np.array(calibration_data['R2'])
    P1 = np.array(calibration_data['P1'])
    P2 = np.array(calibration_data['P2'])
    Q = np.array(calibration_data['Q'])

    return K_left, D_left, K_right, D_right, image_size, R, T, R1, R2, P1, P2, Q


def rectify_images(K_left: np.ndarray, D_left: np.ndarray, K_right: np.ndarray, D_right: np.ndarray,
                   image_size: tuple, R: np.ndarray, T: np.ndarray, img_left: np.ndarray, img_right: np.ndarray) -> tuple:
    """
    Rectify left and right images based on the stereo calibration data.

    Parameters:
        K_left (np.ndarray): Camera matrix for the left camera.
        D_left (np.ndarray): Distortion coefficients for the left camera.
        K_right (np.ndarray): Camera matrix for the right camera.
        D_right (np.ndarray): Distortion coefficients for the right camera.
        image_size (tuple): Size of the input images (width, height).
        R (np.ndarray): Rotation matrix between the two cameras.
        T (np.ndarray): Translation vector between the two cameras.
        img_left (np.ndarray): The left image to be rectified.
        img_right (np.ndarray): The right image to be rectified.

    Returns:
        tuple: Contains the rectified left and right images.
    """
    rectify_scale = 1.0  # Full image area
    (R1, R2, P1, P2, Q, roi1, roi2) = cv2.stereoRectify(
        cameraMatrix1=K_left, distCoeffs1=D_left,
        cameraMatrix2=K_right, distCoeffs2=D_right,
        imageSize=image_size, R=R, T=T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=rectify_scale
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_32FC1)

    img_left_rectified = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    img_right_rectified = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

    return img_left_rectified, img_right_rectified


def compute_disparity(rectified_left: np.ndarray, rectified_right: np.ndarray, image_size: tuple) -> tuple:
    """
    Compute the disparity map using StereoSGBM (Semi-Global Block Matching).

    Parameters:
        rectified_left (np.ndarray): The left rectified image.
        rectified_right (np.ndarray): The right rectified image.
        image_size (tuple): Size of the input images (width, height).

    Returns:
        tuple: Contains the disparity map and the colored disparity map for visualization.
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=200,  # Adjust based on your scene
        blockSize=7,         # Increased block size for better matching
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=2,
        speckleWindowSize=10,
        speckleRange=7,
        preFilterCap=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
    disparity = np.clip(disparity, 0, 255)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_colored = cv2.applyColorMap(disparity_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    return disparity, disparity_colored


def compute_wls_filtered_disparity(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute the WLS (Weighted Least Squares) filtered disparity map.

    Parameters:
        L (np.ndarray): Left rectified image.
        R (np.ndarray): Right rectified image.

    Returns:
        np.ndarray: The WLS filtered disparity map.
    """
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*6,
        blockSize=11,
        P1=8*3*11**2,
        P2=32*3*11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    left_disp = left_matcher.compute(L, R).astype(np.float32) / 16.0
    right_disp = right_matcher.compute(R, L).astype(np.float32) / 16.0

    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(10000); wls.setSigmaColor(1.2)
    filtered = wls.filter(left_disp, L, None, right_disp)
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)

    return filtered


def generate_point_cloud(disparity: np.ndarray, Q: np.ndarray, img_left_rectified: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Generate a 3D point cloud from the disparity map.

    Parameters:
        disparity (np.ndarray): The disparity map.
        Q (np.ndarray): The disparity-to-depth mapping matrix.
        img_left_rectified (np.ndarray): The left rectified image.

    Returns:
        o3d.geometry.PointCloud: The generated 3D point cloud.
    """
    points3d = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > np.percentile(disparity[disparity > 0], 5)
    pts = points3d[mask]
    cols = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2RGB)[mask]

    # Downsample for speed
    step = max(1, len(pts)//100000)
    pts_ds = pts[::step]
    cols_ds = cols[::step]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_ds)
    pcd.colors = o3d.utility.Vector3dVector(cols_ds.astype(np.float64) / 255.0)

    # Flip 180° about X axis
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    pcd.rotate(R, center=(0, 0, 0))

    return pcd


def display_images_and_pcd(img_left_rectified: np.ndarray, img_right_rectified: np.ndarray,
                           disparity_colored: np.ndarray, wls_disp_colored: np.ndarray, pcd: o3d.geometry.PointCloud):
    """
    Display the rectified images, disparity maps, and point cloud in a single matplotlib figure.

    Parameters:
        img_left_rectified (np.ndarray): The left rectified image.
        img_right_rectified (np.ndarray): The right rectified image.
        disparity_colored (np.ndarray): The disparity map for visualization.
        wls_disp_colored (np.ndarray): The WLS filtered disparity map.
        pcd (o3d.geometry.PointCloud): The 3D point cloud to visualize.

    Returns:
        None: Displays the images and point cloud.
    """
    # Create the subplots
    fig = plt.figure(figsize=(15, 15))

    # Display rectified images
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2RGB))
    ax1.set_title("Rectified Left Image")
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2RGB))
    ax2.set_title("Rectified Right Image")
    ax2.axis('off')

    # Display disparity maps
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(cv2.cvtColor(disparity_colored, cv2.COLOR_BGR2RGB))
    ax3.set_title("Disparity Map")
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(cv2.cvtColor(wls_disp_colored, cv2.COLOR_BGR2RGB))
    ax4.set_title("WLS Filtered Disparity")
    ax4.axis('off')

    # Capture the Open3D point cloud visualization as an image
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Hide the window
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Capture the screen
    image = vis.capture_screen_float_buffer(False)
    image = np.asarray(image)
    img_pil = Image.fromarray((image * 255).astype(np.uint8))  # Convert to uint8

    # Display the point cloud in subplot 5
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(img_pil)
    ax5.set_title("Point Cloud")
    ax5.axis('off')

    plt.tight_layout()
    plt.show()

    # Open3D visualization separately
    o3d.visualization.draw_geometries([pcd])


def main():
    """
    Main function to load images, calibrate, rectify, compute disparity, and visualize the results.
    
    Loads stereo calibration data, rectifies images, computes disparity, generates 3D point cloud,
    and visualizes all results in a single window.
    """
    # Load calibration data
    calibration_file = r'../data/stereo_calibration.json'  # Update path if needed
    K_left, D_left, K_right, D_right, image_size, R, T, R1, R2, P1, P2, Q = load_calibration_data(calibration_file)

    # Load images
    img_left = cv2.imread(r"../data/stereo/left.png")
    img_right = cv2.imread(r"../data/stereo/right.png")

    # Rectify images
    img_left_rectified, img_right_rectified = rectify_images(K_left, D_left, K_right, D_right, image_size, R, T, img_left, img_right)

    # Compute disparity
    disparity, disparity_colored = compute_disparity(img_left_rectified, img_right_rectified, image_size)

    # Compute WLS filtered disparity
    wls_disparity = compute_wls_filtered_disparity(img_left_rectified, img_right_rectified)
    wls_disp_colored = cv2.applyColorMap(cv2.normalize(wls_disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)

    # Generate 3D point cloud
    pcd = generate_point_cloud(wls_disparity, Q, img_left_rectified)

    # Display all images and point cloud
    display_images_and_pcd(img_left_rectified, img_right_rectified, disparity_colored, wls_disp_colored, pcd)


if __name__ == "__main__":
    main()
