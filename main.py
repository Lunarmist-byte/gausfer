import os
import argparse
import torch
import torchvision
import torch.optim as optim
torch.backends.cudnn.benchmark = True
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

        self.full_proj=proj@self.w2c

def update_optimizer(old_optimizer, new_param_groups):
    """
    Updates the optimizer with new parameter groups while preserving 
    existing Adam state (momentum/variance) for surviving indices.
    """
    # Create the new optimizer with exactly the same global settings
    new_optimizer = optim.Adam(new_param_groups, lr=0.0, eps=1e-15)
    
    # Transfer state
    for i, new_group in enumerate(new_optimizer.param_groups):
        if i >= len(old_optimizer.param_groups):
            continue
            
        old_group = old_optimizer.param_groups[i]
        
        for old_p, new_p in zip(old_group['params'], new_group['params']):
            if old_p in old_optimizer.state:
                state = old_optimizer.state[old_p]
                new_state = {}
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        if new_p.shape != old_p.shape:
                            new_v = torch.zeros_like(new_p)
                            # Only copy along the first dimension (point index)
                            common_size = min(old_p.shape[0], new_p.shape[0])
                            # Handle different shape ranks (some params might be [N, 1] or [N, 3])
                            if len(v.shape) == 1:
                                new_v[:common_size] = v[:common_size]
                            elif len(v.shape) == 2:
                                new_v[:common_size, :] = v[:common_size, :]
                            elif len(v.shape) == 3:
                                new_v[:common_size, :, :] = v[:common_size, :, :]
                            new_state[k] = new_v
                        else:
                            new_state[k] = v.clone()
                    else:
                        new_state[k] = v
                new_optimizer.state[new_p] = new_state
    return new_optimizer

def main():
    parser = argparse.ArgumentParser(description="Gausfer 3D Reconstruction Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run a quick 5-minute training session")
    parser.add_argument("--resume", action="store_true", help="Resume directly from checkpoint if available")
    args = parser.parse_args()

    print("Starting 3D Reconstruction")
    image_dir='./images'
    output_dir='./output'
    colmap_out = os.path.join(output_dir, 'sparse')
    colmap_model_path = os.path.join(colmap_out, '0')
    #Bridging
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(os.path.join(colmap_model_path, "cameras.txt")):
        print("Fresh images, running COLMAP")
        estimator = PoseEstimator(image_path=image_dir, output_path=output_dir)
        estimator.run_colmap()
    else:
        print("Camera poses already found,skipping colmap")
    #Load dataset
    dataset = RoomDatasetLoader(colmap_dir=colmap_model_path, images_dir=image_dir)
    # Step 1: Calculate scene bounds and load initial points from SfM if available
    print("Seeding Gaussians & Calculating Scene Bounds")
    points_path = os.path.join(colmap_model_path, "points3D.txt")
    init_xyz = None
    init_rgb = None
    bbox = [[-5,-5,-5],[5,5,5]] # Default

    if os.path.exists(points_path):
        print("Reading SfM points from COLMAP...")
        xyz = []
        rgb = []
        with open(points_path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                parts = line.split()
                if len(parts) >= 7:
                    xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    rgb.append([float(parts[4])/255.0, float(parts[5])/255.0, float(parts[6])/255.0])
        
        if xyz:
            init_xyz = torch.tensor(xyz, dtype=torch.float32, device='cuda')
            # Opacity: Start at logit(0.5) = 0.0 for better initial visibility
            opacities = torch.zeros((init_xyz.shape[0], 1), device='cuda')
            init_rgb = torch.tensor(rgb, dtype=torch.float32, device='cuda')
            p_min = torch.quantile(init_xyz, 0.02, dim=0)
            p_max = torch.quantile(init_xyz, 0.98, dim=0)
            bbox = [p_min.tolist(), p_max.tolist()]
            print(f"  Calculated BBox: {bbox}")

    #Initialize NeRF
    nerf=RoomNeRF().cuda()
    
    # Dynamically set near/far planes based on bbox scale
    scene_center = torch.tensor([(bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2, (bbox[0][2]+bbox[1][2])/2])
    scene_radius = torch.norm(torch.tensor([bbox[1][0], bbox[1][1], bbox[1][2]]) - scene_center)
    near = max(0.1, scene_radius.item() * 0.01)
    far = scene_radius.item() * 5.0
    print(f"Setting Trainer Planes: Near={near:.2f}, Far={far:.2f}")

    ckpt_nerf_path = os.path.join(output_dir, "nerf_checkpoint.pth")
    ckpt_gauss_path = os.path.join(output_dir, "gauss_checkpoint.pth")
    gaussians = GaussianModel(sh_degree=3)

    if args.resume and os.path.exists(ckpt_nerf_path) and os.path.exists(ckpt_gauss_path):
        print("Found Checkpoint Data! Resuming directly to Co-Optimization...")
        nerf.load_state_dict(torch.load(ckpt_nerf_path))
        gaussians.load_checkpoint(ckpt_gauss_path)
    else:
        if args.resume:
            print("Resume requested, but no checkpoint found. Starting from scratch...")
            
        trainer=NeRFTrainer(nerf, near=near, far=far)
        #Pre-Train
        print("warming up NeRF")
        num_epochs = 5 if args.quick else 50
        for epoch in range(num_epochs):
            for i in range(len(dataset.images)):
                img,poses=dataset.get_training_batch(i)
                loss=trainer.train_step(img,poses)
                if i % 10 == 0:
                    print(f"  [NeRF] Epoch {epoch} | Iter {i}/{len(dataset.images)} | Loss: {loss:.4f}")
            trainer.step_scheduler() # Decay learning rate
            print(f"NeRF Epoch {epoch} Completed. Final Loss: {loss:.4f}")
        
        # Priority Seeding Strategy
        if args.quick and init_xyz is not None:
            print(f"  Quick mode: Favoring SfM points ({init_xyz.shape[0]}) over NeRF bridge")
            if init_xyz.shape[0] > 100000:
                indices = torch.randperm(init_xyz.shape[0])[:100000]
                init_xyz = init_xyz[indices]
                init_rgb = init_rgb[indices]
            
            # Start at logit(0.5) = 0.0 for better initial visibility in quick mode
            init_opacities = torch.zeros((init_xyz.shape[0], 1), device='cuda')
            gaussians.initialize_from_pcd(init_xyz, init_rgb, opacities=init_opacities)
        else:
            print("  Using Hybrid NeRF-to-Gaussian Bridge...")
            bridge = NeRFToGaussianBridge(nerf, bbox)
            seed_res = 128 if args.quick else 256
            seed_threshold = 0.5
            init_xyz_nerf, init_rgb_nerf = bridge.generate_initial_gaussians(resolution=seed_res, threshold=seed_threshold)
            gaussians.initialize_from_nerf(init_xyz_nerf, init_rgb_nerf)
            
        # Save checkpoints for future resumes
        torch.save(nerf.state_dict(), ckpt_nerf_path)
        gaussians.save_checkpoint(ckpt_gauss_path)
    
    # Calculate scene radius for LR scaling (standard 3DGS practice)
    with torch.no_grad():
        center = gaussians.xyz.mean(dim=0)
        spatial_lr_scale = torch.norm(gaussians.xyz - center, dim=-1).max().item()
        spatial_lr_scale = max(1.0, min(spatial_lr_scale, 20.0))
    print(f"  Scaling parameters to scene radius: {spatial_lr_scale:.2f}")

    # Per-parameter learning rates (critical for 3DGS quality)
    param_groups = [
        {"params": [gaussians._xyz], "lr": 0.00016 * spatial_lr_scale, "name": "xyz"},
        {"params": [gaussians._features_dc], "lr": 0.0025, "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": 0.000125, "name": "f_rest"},
        {"params": [gaussians._opacity], "lr": 0.05, "name": "opacity"},
        {"params": [gaussians._scaling], "lr": 0.005, "name": "scaling"},
        {"params": [gaussians._rotation], "lr": 0.001, "name": "rotation"},
    ]
    gaussians_optim = optim.Adam(param_groups, lr=0.0, eps=1e-15)
    
    #Optimization Loop
    print("Optimizing...")
    co_optimizer=HybridCoOptimizer(nerf,gaussians)
    rasterizer=RoomRasterizerCUDA()

    num_steps = 500 if args.quick else 5000
    for step in range(num_steps):
        # Exponential LR decay for positions (critical for high PSNR convergence)
        progress = step / num_steps
        current_xyz_lr = (0.00016 * spatial_lr_scale) * ((0.01) ** progress)
        for param_group in gaussians_optim.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = current_xyz_lr

        gaussians_optim.zero_grad()
        #fetch sequentially
        idx=step%len(dataset.images)
        ground_truth_image,(H,W,K,c2w)=dataset.get_training_batch(idx)
        #Wrap the NeRF matrices into Camera
        view_cam=ViewCamera(H,W,K,c2w)
        old_count = gaussians.xyz.shape[0]
        loss, rendered_image = co_optimizer.step(view_cam=view_cam,ground_truth_image=ground_truth_image,render_func=rasterizer.render_room_view, step=step)
        
        if gaussians.xyz.shape[0] != old_count:
            # Rebuild optimizer with per-parameter LRs while preserving Adam state
            param_groups = [
                {"params": [gaussians._xyz], "lr": current_xyz_lr, "name": "xyz"},
                {"params": [gaussians._features_dc], "lr": 0.0025, "name": "f_dc"},
                {"params": [gaussians._features_rest], "lr": 0.000125, "name": "f_rest"},
                {"params": [gaussians._opacity], "lr": 0.05, "name": "opacity"},
                {"params": [gaussians._scaling], "lr": 0.005, "name": "scaling"},
                {"params": [gaussians._rotation], "lr": 0.001, "name": "rotation"},
            ]
            gaussians_optim = update_optimizer(gaussians_optim, param_groups)

        #Apply graident updates
        gaussians_optim.step()
        
        if step % 10 == 0:
            image_filename = dataset.images[idx]['name']
            print(f"Step {step}/{num_steps} | Loss: {loss:.4f} | View {idx} ({image_filename}) | Splats: {gaussians.xyz.shape[0]} | SH: {gaussians.active_sh_degree}")

    print("\n Full Room Reconstruction done")
    
    # Save Final Result
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "result.png")
    torchvision.utils.save_image(rendered_image.permute(2,0,1), result_path)
    print(f"Final result saved to {result_path}")
if __name__=="__main__":
    main()




