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
