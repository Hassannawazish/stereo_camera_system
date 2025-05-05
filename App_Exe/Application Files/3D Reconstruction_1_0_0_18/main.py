import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
import os


def load_calibration_data(file_path: str) -> tuple:
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    with open(file_path, 'r') as f:
        calibration_data = json.load(f)

    K_left = np.array(calibration_data['K_left'])
    D_left = np.array(calibration_data['D_left'])
    K_right = np.array(calibration_data['K_right'])
    D_right = np.array(calibration_data['D_right'])
    image_size = tuple(calibration_data['image_size'])

    R = np.eye(3)
    T = np.array([[-0.1], [0.0], [0.0]])
    R1 = np.array(calibration_data['R1'])
    R2 = np.array(calibration_data['R2'])
    P1 = np.array(calibration_data['P1'])
    P2 = np.array(calibration_data['P2'])
    Q = np.array(calibration_data['Q'])

    return K_left, D_left, K_right, D_right, image_size, R, T, R1, R2, P1, P2, Q


def rectify_images(K_left, D_left, K_right, D_right, image_size, R, T, img_left, img_right):
    rectify_scale = 1.0
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=rectify_scale
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_32FC1)

    img_left_rectified = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    img_right_rectified = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

    return img_left_rectified, img_right_rectified


def compute_disparity(rectified_left, rectified_right, image_size):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=200,
        blockSize=7,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
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


def compute_wls_filtered_disparity(L, R):
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 6,
        blockSize=11,
        P1=8 * 3 * 11 ** 2,
        P2=32 * 3 * 11 ** 2,
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
    wls.setLambda(10000)
    wls.setSigmaColor(1.2)
    filtered = wls.filter(left_disp, L, None, right_disp)
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)

    return filtered


def generate_point_cloud(disparity, Q, img_left_rectified):
    points3d = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > np.percentile(disparity[disparity > 0], 5)
    pts = points3d[mask]
    cols = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2RGB)[mask]

    step = max(1, len(pts) // 100000)
    pts_ds = pts[::step]
    cols_ds = cols[::step]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_ds)
    pcd.colors = o3d.utility.Vector3dVector(cols_ds.astype(np.float64) / 255.0)

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    pcd.rotate(R, center=(0, 0, 0))

    return pcd


def display_images_and_pcd(img_left_rectified, img_right_rectified, disparity_colored, wls_disp_colored, pcd):
    fig = plt.figure(figsize=(15, 15))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2RGB))
    ax1.set_title("Rectified Left Image")
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2RGB))
    ax2.set_title("Rectified Right Image")
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(cv2.cvtColor(disparity_colored, cv2.COLOR_BGR2RGB))
    ax3.set_title("Disparity Map")
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(cv2.cvtColor(wls_disp_colored, cv2.COLOR_BGR2RGB))
    ax4.set_title("WLS Filtered Disparity")
    ax4.axis('off')

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(False)
    vis.destroy_window()

    img_pil = Image.fromarray((np.asarray(image) * 255).astype(np.uint8))
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(img_pil)
    ax5.set_title("Point Cloud")
    ax5.axis('off')

    plt.tight_layout()
    plt.show()

    o3d.visualization.draw_geometries([pcd])


def main():
    # Robust path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_file = os.path.join(script_dir, "stereo_calibration.json")
    left_img_path = os.path.join(script_dir, "stereo", "left.jpg")
    right_img_path = os.path.join(script_dir, "stereo", "right.jpg")

    # Load calibration and images
    K_left, D_left, K_right, D_right, image_size, R, T, R1, R2, P1, P2, Q = load_calibration_data(calibration_file)
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)

    if img_left is None or img_right is None:
        print("Error: One or both stereo images not found.")
        return

    img_left_rectified, img_right_rectified = rectify_images(K_left, D_left, K_right, D_right, image_size, R, T, img_left, img_right)
    disparity, disparity_colored = compute_disparity(img_left_rectified, img_right_rectified, image_size)
    wls_disparity = compute_wls_filtered_disparity(img_left_rectified, img_right_rectified)
    wls_disp_colored = cv2.applyColorMap(cv2.normalize(wls_disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    pcd = generate_point_cloud(wls_disparity, Q, img_left_rectified)
    display_images_and_pcd(img_left_rectified, img_right_rectified, disparity_colored, wls_disp_colored, pcd)


if __name__ == "__main__":
    main()
