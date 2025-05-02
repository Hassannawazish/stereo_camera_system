# Stereo Vision System

![Stereo Vision System]

## 📌 Project Overview

The **Stereo Vision System** is a software-based 3D geometry reconstruction solution using two USB cameras connected to a standard computer. The goal is to capture stereo image pairs, process them, and generate a **3D point cloud**. The system is modular, with key components for camera calibration, stereo rectification, feature detection, geometry estimation, triangulation, and post-processing.

## 🎯 Key Features

- **Camera Calibration**: Computes intrinsic and extrinsic camera parameters.
- **Stereo Rectification**: Aligns stereo images for better depth estimation.
- **Feature Detection & Matching**: Finds corresponding points in stereo images.
- **Stereo Geometry Estimation**: Computes essential matrix and camera poses.
- **3D Triangulation & Reconstruction**: Generates sparse 3D point clouds.
- **Point Cloud Post-Processing**: Filters and enhances the 3D reconstruction.

---

## 🛠 Installation & Setup

### **1️⃣ Prerequisites**
Ensure you have the following dependencies installed:

- **OS**: Windows / Linux / MacOS
- **Python (Recommended: 3.8+)**
- **Libraries**:
  ```bash
  pip install numpy opencv-python opencv-contrib-python matplotlib scipy scikit-image

## Camera Callibration using checkerboard images and dump all parameters into json file.
- Camera calibration for both left and right cameras using a checkerboard pattern, computes the intrinsic parameters of each camera, and then performs stereo calibration to align the two cameras.
- Calibration data is saved in a JSON file stereo_calibration.json.
- Stereo calibration computes:

    Rotation Matrix (R): The rotation between the two cameras.

    Translation Vector (T): The translation between the two cameras.

    Essential Matrix (E): Describes the relation between corresponding points in the two images.

    Fundamental Matrix (F): Relates points between the stereo pair of images.

## Stereo Rectification and visualization of Epipolar lines.
- Stereo Calibration parameters are used for stereo rectification and correct epipolar lines are drawn using OpenCV functions.

