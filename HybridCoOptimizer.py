import torch
from loss_utils import ssim

class HybridCoOptimizer:
    '''implements nerf directly to gaussian
    uses volumetric understanding of NeRF to guide the densification
    and pruning of 3DGS'''
    def __init__(self,nerf_model,gaussian_model,grad_threshold=0.0002,density_threshold=20.0, lambda_dssim=0.2):
        self.nerf=nerf_model
        self.gaussians=gaussian_model
        self.grad_threshold=grad_threshold
        self.density_threshold=density_threshold
        self.lambda_dssim=lambda_dssim
    def step(self,view_cam,ground_truth_image,render_func, step=0):
        #render the gaussians
        render_pkg=render_func(view_cam,self.gaussians)
        rendered_image=render_pkg["render"].permute(1,2,0) # Change [3,H,W] to [H,W,3]
        
        #calculate rasterization loss
        l1_loss=torch.nn.functional.l1_loss(rendered_image,ground_truth_image)
        
        # calculate ssim loss
        # ssim expects [B, C, H, W]
        img1 = rendered_image.permute(2,0,1).unsqueeze(0)
        img2 = ground_truth_image.permute(2,0,1).unsqueeze(0)
        ssim_loss = 1.0 - ssim(img1, img2)
        
        loss = (1.0 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss
        loss.backward()
        
        # Periodic NeRF Guided Densification/Pruning (expensive, so only every 50 steps)
        if step > 0 and step % 50 == 0:
            #Nerf Guided Densification
            with torch.no_grad():
                grads=self.gaussians.xyz.grad
                if grads is not None:
                    high_error_mask=torch.norm(grads,dim=-1)>self.grad_threshold
                    if high_error_mask.any():
                        problematic_xyz=self.gaussians.xyz[high_error_mask]
                        #asking if nerf detects solid matter
                        nerf_density=torch.relu(self.nerf(problematic_xyz)[...,3])
                        valid_split_mask=nerf_density>self.density_threshold

                        if valid_split_mask.any():
                            combined_mask=torch.zeros_like(high_error_mask)
                            combined_mask[high_error_mask]=valid_split_mask
                            
                            max_scales = torch.max(self.gaussians.scales, dim=1).values
                            # Scene extent is around 10 based on bbox [-5, 5]. Let's say 0.01 * 10 = 0.1
                            split_mask = combined_mask & (max_scales > 0.1)
                            clone_mask = combined_mask & (max_scales <= 0.1)
                            
                            self.gaussians.densify_clone_split(clone_mask, split_mask)

            #nerf-guided pruning
            with torch.no_grad():
                current_density=torch.relu(self.nerf(self.gaussians.xyz)[...,3])
                prune_mask = (self.gaussians.opacity.squeeze() < 0.01)
                
                # Also prune Gaussians that are way too large
                max_scales = torch.max(self.gaussians.scales, dim=1).values
                prune_mask = prune_mask | (max_scales > 2.0)

                print(f"DEBUG: Pruning {prune_mask.sum().item()} points out of {self.gaussians.xyz.shape[0]}")
                if prune_mask.any():
                    self.gaussians.prune_points(prune_mask)
                    print(f"DEBUG: Splat count is now {self.gaussians.xyz.shape[0]}")
        
        return loss.item(), rendered_image

