import os
import torch
import torch.optim as optim

from pose_estimator import PoseEstimator
from dataset_loader import RoomDatasetLoader
from nerf_model import RoomNeRF
from nerf_trainer import NeRFTrainer
from bridge_converter import NeRFToGaussianBridge
from HybridCoOptimizer import HybridCoOptimizer
from rasterizer import RoomRasterizerCUDA
from gaussian_model import GaussianModel

class ViewCamera:
    '''Helper class to translate NeRF to 3DGS'''
    def __init__(self,H,W,K,c2w):
        self.H=H
        self.W=W
        fx,fy=K[0,0].item(),K[1,1].item()
        #Calculate field of view
        self.tanfovx=W/(2.0*fx)
        self.tanfovy=H/(2.0*fy)
        #Rasterizer w2c
        self.w2c=torch.inverse(c2w)
        self.pos=c2w[:3,3]

        #Build standard OpenGL projection matrix
        znear,zfar=0.01,100.0
        proj=torch.zeros((4,4),device='cuda')
        proj[0,0]=2.0*fx/W
        proj[1,1]=2.0*fy/H
        proj[2,2]=zfar/(zfar-znear)
        proj[2,3]=-(zfar*znear)/(zfar-znear)
        proj[3,2]=1.0

        self.full_proj=self.w2c@proj


def main():
    print("Starting 3D Reconstruction")
    image_dir='./images'
    output_dir='./output'
    colmap_out = os.path.join(output_dir, 'sparse')
    colmap_model_path = os.path.join(colmap_out, '0')
    #Bridging
    if not os.path.exists(os.path.join(colmap_model_path, "cameras.txt")):
        print("Fresh images, running COLMAP")
        estimator = PoseEstimator(image_path=image_dir, output_path=output_dir)
        estimator.run_colmap()
    else:
        print("Camera poses already found,skipping colmap")
    #Load dataset
    dataset = RoomDatasetLoader(colmap_dir=colmap_model_path, images_dir=image_dir)
    #Initialize NeRFrde 
    nerf=RoomNeRF().cuda()
    trainer=NeRFTrainer(nerf)
    #Pre-Train
    print("warming up NeRF")
    for epoch in range(10):
        for i in range(len(dataset.images)):
            img,poses=dataset.get_training_batch(i)
            loss=trainer.train_step(img,poses)
        print(f"NeRF Epoch {epoch} Loss:{loss:.4f}")
    #Bridge
    print("Seeding Gaussians")
    bbox=[[-5,-5,-5],[5,5,5]]#example bounds
    bridge=NeRFToGaussianBridge(nerf,bbox)
    init_xyz,init_rgb=bridge.generate_initial_gaussians()
    #initalizing space for gaussians
    gaussians=GaussianModel(sh_degree=3)
    gaussians.initialize_from_nerf(init_xyz,init_rgb)
    #Co-Optimizer
    gaussians_optim=optim.Adam(gaussians.parameters(),lr=0.001)
    print("Optimizing....")
    co_optimizer=HybridCoOptimizer(nerf,gaussians)
    rasterizer=RoomRasterizerCUDA()

    num_steps=5000
    for step in range(num_steps):
        gaussians_optim.zero_grad()
        #fetch sequentially
        idx=step%len(dataset.images)
        ground_truth_image,(H,W,K,c2w)=dataset.get_training_batch(idx)
        #Wrap the NeRF matrices into Camera
        view_cam=ViewCamera(H,W,K,c2w)
        old_count = gaussians.xyz.shape[0]
        loss=co_optimizer.step(view_cam=view_cam,ground_truth_image=ground_truth_image,render_func=rasterizer.render_room_view)
        
        if gaussians.xyz.shape[0] != old_count:
            gaussians_optim = optim.Adam(gaussians.parameters(), lr=0.001)

        #Apply graident updates
        gaussians_optim.step()
        
        if step%100==0:
            print(f"step{step}/{num_steps}| Loss:{loss:.4f}|Splat Count:{gaussians.xyz.shape[0]}")

    print("\n Full Room Reconstruction done")
if __name__=="__main__":
    main()




