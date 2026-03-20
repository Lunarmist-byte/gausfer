# Hybrid Neural Rendering Pipeline: NeRF to 3D Gaussian Splatting

This repository implements a high-fidelity 3D reconstruction pipeline that bridges Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). The project focuses on utilizing the volumetric consistency of NeRF to initialize and refine high-density Gaussian primitives for real-time, room-scale rendering.

## Key Features

- **Automated Pose Estimation**: Integrated COLMAP wrapper for Structure-from-Motion (SfM) to extract camera intrinsics and extrinsics.
- **Volumetric Scene Representation**: A deep MLP-based NeRF implementation for predicting density and RGB from 3D coordinates.
- **Hybrid Transition Bridge**: A sampling module that converts NeRF density fields into point clouds to initialize Gaussian Splats.
- **Dynamic Gaussian Optimization**: Adaptive densification and pruning logic to handle complex geometry and improve rendering efficiency.
- **Desktop Application Interface (PyQt6)**: A multithreaded GUI (`app_ui.py`) for managing video extraction, starting the training pipeline, and viewing real-time rendering logs.
- **Intelligent Video Extraction**: A Streamlit tool (`data_capture_app.py`) for automated frame extraction, leveraging Laplacian variance to filter out blurry frames automatically.

## Project Structure

### 1. Camera Pose Estimation (PoseEstimator)
Uses COLMAP to perform feature extraction and exhaustive matching. This step is critical for recovering the camera parameters required for both NeRF training and Gaussian projection.

### 2. Neural Radiance Field (RoomNeRF)
A volumetric representation using a PyTorch-based MLP. Inputs: 5D coordinates (location x, y, z and viewing direction). Outputs: Volume density ($\sigma$) and emitted RGB color. Managed by `NeRFTrainer`.

### 3. NeRF-to-Gaussian Bridge (NeRFToGaussianBridge)
This module implements the transition between representations. By sampling the NeRF model's density field within a defined bounding box, it identifies high-probability surfaces to serve as the "initial seeds" for 3D Gaussians.

### 4. Gaussian Optimization (HybridCoOptimizer)
Implements the "Densify and Prune" strategy to scale reconstruction to full rooms. Includes advanced rendering passes via `diff-gaussian-rasterization`.

### 5. Desktop Controller (app_ui.py)
A native desktop widget built with PyQt6. This acts as the primary access point for the Gausfer pipeline, running heavy PyTorch and COLMAP computations in background threads keeping the UI highly responsive.

### 6. Pipeline Feeder (data_capture_app.py)
A Streamlit web application for users to upload room walkthrough videos and automatically curate sharp frames based on blur variance.

## Getting Started

### Prerequisites

Python 3.8+
PyTorch (CUDA supported)
COLMAP installed and added to your system PATH.
PyQt6 & Streamlit (for GUI tools)

### Installation
```bash
git clone https://github.com/Lunarmist-byte/gausfer
cd gausfer
# 1. Create venv
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Install rasterizer (IMPORTANT)
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```
OR Run
```bash
setup.bat #windows
setup.sh  #linux
```
## System Requirements
### Linux
CUDA 12.1
nvcc available (nvcc --version)
GCC / G++ installed
### Windows
CUDA 12.1
Visual Studio Build Tools
→ “Desktop development with C++
## Usage Workflow

You can now run the full pipeline either using our new Desktop UI or the command line.

### Option A: Desktop UI (Recommended)
Launch the integrated desktop application to manage the entire workflow via a graphical interface:
```bash
python app_ui.py
```
From here you can:
1. Extract sharp frames from a video.
2. Monitor COLMAP extraction, NeRF warmup, Bridging, and 3DGS progression with live logs.

### Option B: Command Line

**1. Data Preparation:**
Use the Streamlit app to extract frames from a video or place your images in `./images`.
```bash
streamlit run data_capture_app.py
```

**2. Run Pipeline:**
Run the core pipeline end-to-end using `main.py`:
```bash
python main.py
```
This handles COLMAP extraction, NeRF training, Bridging & Splatting, and Refinement processes seamlessly in the terminal.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Lunarmist-byte/gausfer/blob/main/license.pdf) file for details.

## Acknowledgments

- Original NeRF Paper (Mildenhall et al.)
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al.)
