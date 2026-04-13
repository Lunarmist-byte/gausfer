import torch
from loss_utils import ssim

class HybridCoOptimizer:
    '''implements nerf directly to gaussian
    uses volumetric understanding of NeRF to guide the densification
    and pruning of 3DGS'''
    def __init__(self,nerf_model,gaussian_model,grad_threshold=0.0002,density_threshold=10.0, prune_density_threshold=0.5, lambda_dssim=0.2):
        self.nerf=nerf_model
        self.gaussians=gaussian_model
        self.grad_threshold=grad_threshold
        self.density_threshold=density_threshold
        self.prune_density_threshold=prune_density_threshold
        self.lambda_dssim=lambda_dssim
        self.min_points=1000 # Never prune below this count
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
        
        # SH Degree Warmup: increase every 1000 steps for gradual color complexity
        if step > 0 and step % 1000 == 0:
            self.gaussians.increase_sh_degree()
        
        # Densification/Pruning: only after step 500, every 100 steps
        # This gives Gaussians time to settle before reshaping
        if step >= 500 and step % 100 == 0:
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
                            # Tighter splitting threshold for room detail
                            split_mask = combined_mask & (max_scales > 0.05)
                            clone_mask = combined_mask & (max_scales <= 0.05)
                            
                            n_clone = clone_mask.sum().item()
                            n_split = split_mask.sum().item()
                            if n_clone > 0 or n_split > 0:
                                print(f"  Densification: Clones: {n_clone}, Splits: {n_split}")
                                self.gaussians.densify_clone_split(clone_mask, split_mask)

            #nerf-guided pruning (conservative)
            with torch.no_grad():
                current_count = self.gaussians.xyz.shape[0]
                
                # Skip pruning if already at minimum
                if current_count > self.min_points:
                    # Chunked NeRF evaluation for pruning
                    nerf_densities = []
                    gauss_xyz = self.gaussians.xyz
                    chunk_size = 65536
                    for i in range(0, gauss_xyz.shape[0], chunk_size):
                        chunk = gauss_xyz[i:i+chunk_size]
                        nerf_densities.append(torch.relu(self.nerf(chunk)[...,3]))
                    current_density = torch.cat(nerf_densities, dim=0)
                    
                    # Softer opacity threshold — don't kill semi-transparent points too early
                    opacity_mask = (self.gaussians.opacity.squeeze() < 0.005)
                    
                    # Prune Gaussians that are way too large (blobs)
                    max_scales = torch.max(self.gaussians.scales, dim=1).values
                    size_mask = (max_scales > 0.5)

                    # NeRF empty-space pruning (relaxed threshold)
                    nerf_mask = (current_density < self.prune_density_threshold)

                    prune_mask = opacity_mask | size_mask | nerf_mask
                    
                    # Enforce minimum point count
                    n_prune = prune_mask.sum().item()
                    if n_prune > 0 and (current_count - n_prune) >= self.min_points:
                        print(f"  Pruning {n_prune} points (Opacity: {opacity_mask.sum().item()}, Size: {size_mask.sum().item()}, NeRF: {nerf_mask.sum().item()})")
                        self.gaussians.prune_points(prune_mask)
                        print(f"  Splat count: {self.gaussians.xyz.shape[0]}")
                    elif n_prune > 0:
                        # Would go below min, only prune the most egregious (opacity dead)
                        safe_prune = opacity_mask & size_mask  # Only prune truly dead blobs
                        n_safe = safe_prune.sum().item()
                        if n_safe > 0 and (current_count - n_safe) >= self.min_points:
                            print(f"  Safe pruning {n_safe} dead blobs (below min-point guard)")
                            self.gaussians.prune_points(safe_prune)
        
        return loss.item(), rendered_image


