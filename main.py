import torch
from pose_estimator import PoseEstimator
from dataset_loader import RoomDatasetLoader
from nerf_model import RoomNeRF
from nerf_trainer import NeRFTrainer
from bridge_converter import NeRFToGaussianBridge
from HybridCoOptimizer import HybridCoOptimizer
from rasterizer import RoomRasterizerCUDA

def main():
    print("Starting 3D Reconstruction")
    #Run Pose Estimation(COLMAP)
    estimator=PoseEstimator(image_path='./images',output_path='./output')
    estimator.run_colmap()
    #Load dataset
    dataset=RoomDatasetLoader(colmap_dir="./output/sparse",images_dir="./images")
    #Initialize NeRF
    nerf=RoomNeRF().cuda()
    trainer=NeRFTrainer(nerf)
    #Pre-Train
    print("warming up NeRF")