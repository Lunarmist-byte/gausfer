import torch
from loss_utils import ssim

class HybridCoOptimizer:
    '''implements nerf directly to gaussian
    uses volumetric understanding of NeRF to guide the densification
    and pruning of 3DGS'''
    def __init__(self,nerf_model,gaussian_model,grad_threshold=0.0001,density_threshold=20.0, prune_density_threshold=2.0, lambda_dssim=0.2):
        self.nerf=nerf_model
        self.gaussians=gaussian_model
        self.grad_threshold=grad_threshold
        self.density_threshold=density_threshold
        self.prune_density_threshold=prune_density_threshold
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
                        
                        # Chunked NeRF evaluation to prevent OOM/Stalls
                        nerf_densities = []
                        chunk_size = 65536
                        for i in range(0, problematic_xyz.shape[0], chunk_size):
                            chunk = problematic_xyz[i:i+chunk_size]
                            nerf_densities.append(torch.relu(self.nerf(chunk)[...,3]))
                        nerf_density = torch.cat(nerf_densities, dim=0)
                        
                        valid_split_mask=nerf_density>self.density_threshold

                        if valid_split_mask.any():
                            combined_mask=torch.zeros_like(high_error_mask)
                            combined_mask[high_error_mask]=valid_split_mask
                            
                            max_scales = torch.max(self.gaussians.scales, dim=1).values
                            # Scene extent is around 10 based on bbox [-5, 5]. Let's say 0.01 * 10 = 0.1
                            split_mask = combined_mask & (max_scales > 0.1)
                            clone_mask = combined_mask & (max_scales <= 0.1)
                            
                            print(f"DEBUG: Co-Optimization triggering densification: Clones: {clone_mask.sum().item()}, Splits: {split_mask.sum().item()}")
                            self.gaussians.densify_clone_split(clone_mask, split_mask)

            #nerf-guided pruning
            with torch.no_grad():
                # Chunked NeRF evaluation for pruning too
                nerf_densities = []
                gauss_xyz = self.gaussians.xyz
                for i in range(0, gauss_xyz.shape[0], chunk_size):
                    chunk = gauss_xyz[i:i+chunk_size]
                    nerf_densities.append(torch.relu(self.nerf(chunk)[...,3]))
                current_density = torch.cat(nerf_densities, dim=0)
                
                opacity_mask = (self.gaussians.opacity.squeeze() < 0.01)
                
                # Also prune Gaussians that are way too large
                max_scales = torch.max(self.gaussians.scales, dim=1).values
                size_mask = (max_scales > 2.0)

                # Actually use NeRF to prune empty space
                nerf_mask = (current_density < self.prune_density_threshold)

                prune_mask = opacity_mask | size_mask | nerf_mask

                if prune_mask.any():
                    p_opac = opacity_mask.sum().item()
                    p_size = size_mask.sum().item()
                    p_nerf = nerf_mask.sum().item()
                    print(f"DEBUG: Pruning {prune_mask.sum().item()} points (Opacity: {p_opac}, Size: {p_size}, NeRF: {p_nerf})")
                    self.gaussians.prune_points(prune_mask)
                    print(f"DEBUG: Splat count reached {self.gaussians.xyz.shape[0]}")
                else:
                    min_dens = current_density.min().item()
                    near_opacity = (self.gaussians.opacity.squeeze() < 0.05).sum().item()
                    print(f"DEBUG: Pruning cycle skipped (0 points met criteria). Min Density: {min_dens:.4f}, Points near opacity threshold (<0.05): {near_opacity}")
        
        return loss.item(), rendered_image

