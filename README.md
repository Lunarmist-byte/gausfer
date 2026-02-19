Hybrid Neural Rendering Pipeline: NeRF to 3D Gaussian Splatting

This repository implements a high-fidelity 3D reconstruction pipeline that bridges Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). The project focuses on utilizing the volumetric consistency of NeRF to initialize and refine high-density Gaussian primitives for real-time, room-scale rendering.

 Key Features

Automated Pose Estimation: Integrated COLMAP wrapper for Structure-from-Motion (SfM) to extract camera intrinsics and extrinsics.

Volumetric Scene Representation: A deep MLP-based NeRF implementation for predicting density and RGB from 3D coordinates.

Hybrid Transition Bridge: A sampling module that converts NeRF density fields into point clouds to initialize Gaussian Splats.

Dynamic Gaussian Optimization: Adaptive densification and pruning logic to handle complex geometry and improve rendering efficiency.

 Project Structure

1. Camera Pose Estimation (PoseEstimator)

Uses COLMAP to perform feature extraction and exhaustive matching. This step is critical for recovering the camera parameters required for both NeRF training and Gaussian projection.

Features: Database management, SfM mapping, and sparse reconstruction export.

2. Neural Radiance Field (NeRFNetwork)

A volumetric representation using a PyTorch-based MLP.

Inputs: 5D coordinates (location $x, y, z$ and viewing direction $\theta, \phi$).

Outputs: Volume density ($\sigma$) and emitted RGB color.

Trainer: NeRFTrainer handles the GPU-accelerated optimization loop using ray-tracing volumetric rendering.

3. NeRF-to-Gaussian Bridge (NeRFToGaussianBridge)

This module implements the transition between representations. By sampling the NeRF model's density field within a defined bounding box, it identifies high-probability surfaces to serve as the "initial seeds" for 3D Gaussians.

Process: Grid sampling $\rightarrow$ Density Thresholding $\rightarrow$ Color Querying $\rightarrow$ Gaussian Initialization.

4. Gaussian Optimization (GaussianOptimizer)

Implements the "Densify and Prune" strategy to scale reconstruction to full rooms.

Densification: Splits Gaussians in areas with high positional gradients.

Pruning: Removes Gaussians with opacity below a specific threshold ($\alpha < 0.01$) to maintain a clean, efficient model.

 Getting Started

Prerequisites

Python 3.8+

PyTorch (CUDA supported)

COLMAP installed and added to your system PATH.

Installation
```bash
git clone [https://github.com/Lunarmist-byte/gausfer
cd gausfer
pip install -r requirements.txt
```


Usage Workflow

Extract Poses:

estimator = PoseEstimator(image_path="./data/room", output_path="./output")
estimator.run_colmap()



Train NeRF:
Initialize the NeRFNetwork and run the NeRFTrainer to learn the volumetric geometry.

Bridge & Splat:
Use the NeRFToGaussianBridge to sample the trained NeRF and initialize your Gaussian model.

Refine:
Run the GaussianOptimizer to perform final densification and pruning for real-time performance.

 Technical Specifications

Component

Logic

Optimizer

Adam (for both NeRF and GS)

Densification Threshold

$0.0002$ (Gradient-based)

Density Threshold

$15.0$ (for NeRF sampling)

Activation

ReLU for Density, Sigmoid for RGB

 License

This project is licensed under the MIT License - see the LICENSE file for details.

 Acknowledgments

Original NeRF Paper (Mildenhall et al.)

3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al.)
