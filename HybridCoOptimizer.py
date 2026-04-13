import torch
from loss_utils import ssim

class HybridCoOptimizer:
    '''implements nerf directly to gaussian
    uses volumetric understanding of NeRF to guide the densification
    and pruning of 3DGS'''
    def __init__(self,nerf_model,gaussian_model,grad_threshold=0.0001,density_threshold=0.5, lambda_dssim=0.2):
        self.nerf=nerf_model
        self.gaussians=gaussian_model
        # Gradient阈值 (tau_xyz in 3DGS paper)
        self.grad_threshold=grad_threshold
        self.density_threshold=density_threshold
        self.lambda_dssim=lambda_dssim
        self.min_points=1000 # Never prune below this count
        
        # State for gradient accumulation (3DGS style)
        self.xyz_gradient_accum = torch.zeros((self.gaussians.xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.gaussians.xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros((self.gaussians.xyz.shape[0]), device='cuda')

    def step(self,view_cam,ground_truth_image,render_func, step=0):
        #render the gaussians
        render_pkg=render_func(view_cam,self.gaussians)
        rendered_image=render_pkg["render"].permute(1,2,0) # Change [3,H,W] to [H,W,3]
        
        # We need the 2D view-space coordinates and gradients for 3DGS style densification
        # The rasterizer provides these in render_pkg
        viewspace_points = render_pkg["viewspace_points"]
        radii = render_pkg["radii"]
        
        #calculate rasterization loss
        l1_loss = torch.nn.functional.l1_loss(rendered_image, ground_truth_image)
        
        # calculate ssim loss
        # ssim expects [B, C, H, W]
        img1 = rendered_image.permute(2,0,1).unsqueeze(0)
        img2 = ground_truth_image.permute(2,0,1).unsqueeze(0)
        ssim_loss = 1.0 - ssim(img1, img2)
        
        # Ensure losses are scalars (handles accidental broadcasting)
        if l1_loss.dim() > 0: l1_loss = l1_loss.mean()
        if ssim_loss.dim() > 0: ssim_loss = ssim_loss.mean()

        loss = (1.0 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss
        loss.backward()
        
        # 3DGS-style Gradient Accumulation & State Management
        with torch.no_grad():
            # Expansion Guard: Ensure buffers always match point count, even if no grad this step
            if self.xyz_gradient_accum.shape[0] != self.gaussians.xyz.shape[0]:
                new_count = self.gaussians.xyz.shape[0]
                old_count = self.xyz_gradient_accum.shape[0]
                
                new_accum = torch.zeros((new_count, 1), device='cuda')
                new_denom = torch.zeros((new_count, 1), device='cuda')
                new_max_radii = torch.zeros((new_count), device='cuda')
                
                min_len = min(old_count, new_count)
                new_accum[:min_len] = self.xyz_gradient_accum[:min_len]
                new_denom[:min_len] = self.denom[:min_len]
                new_max_radii[:min_len] = self.max_radii2D[:min_len]
                
                self.xyz_gradient_accum = new_accum
                self.denom = new_denom
                self.max_radii2D = new_max_radii

            # Radii tells us which points were actually visible/rendered
            visible_mask = radii > 0
            
            # Use viewspace gradients for more accurate image-space error capture
            if viewspace_points.grad is not None:
                # Norm of 2D gradients in screen space
                v_grads = viewspace_points.grad[visible_mask]
                v_grads_norm = torch.norm(v_grads[:, :2], dim=-1, keepdim=True)
                
                # Gradient Clamping to prevent NaNs
                v_grads_norm = torch.clamp(v_grads_norm, max=0.1)

                self.xyz_gradient_accum[visible_mask] += v_grads_norm
                self.denom[visible_mask] += 1
                self.max_radii2D[visible_mask] = torch.max(self.max_radii2D[visible_mask], radii[visible_mask].float())

        # SH Degree Warmup: increase every 1000 steps for gradual color complexity
        if step > 0 and step % 1000 == 0:
            self.gaussians.increase_sh_degree()
        
        # Opacity Reset: force Gaussians to re-earn visibility
        # Critical for PSNR to remove floaters
        if step > 0 and step % 3000 == 0:
            with torch.no_grad():
                print(f"  Opacity reset at step {step} -- removing floaters")
                # Reset to 0.01 (sigmoid domain is -4.59)
                self.gaussians._opacity.data.fill_(torch.logit(torch.tensor(0.01)).item())
        
        # Densification/Pruning: every 100 steps
        if step >= 500 and step % 100 == 0:
            with torch.no_grad():
                # Average accumulated gradients with epsilon for numerical stability
                avg_grads = self.xyz_gradient_accum / torch.clamp(self.denom, min=1.0)
                # Cleanup any potential NaNs in the gradient accumulation
                avg_grads = torch.nan_to_num(avg_grads)
                high_error_mask = avg_grads.squeeze() > self.grad_threshold
                
                if high_error_mask.any():
                    max_scales = torch.max(self.gaussians.scales, dim=1).values
                    # 3DGS Paper Rule: Split large ones, clone small ones
                    # Using world-space scale comparison (was 0.01 - too aggressive, now 0.05)
                    split_mask = high_error_mask & (max_scales > 0.05)
                    clone_mask = high_error_mask & (max_scales <= 0.05)
                    
                    # Hard cap: never create more than 5000 new splats per densification step
                    # Too many at once = VRAM spike = CUDA crash
                    MAX_NEW_PER_STEP = 5000
                    if split_mask.sum() + clone_mask.sum() > MAX_NEW_PER_STEP:
                        # Prioritize the highest-error points
                        combined = high_error_mask.nonzero(as_tuple=True)[0]
                        top_indices = avg_grads.squeeze()[combined].topk(MAX_NEW_PER_STEP).indices
                        selected = combined[top_indices]
                        new_split = torch.zeros_like(split_mask)
                        new_clone = torch.zeros_like(clone_mask)
                        new_split[selected] = split_mask[selected]
                        new_clone[selected] = clone_mask[selected]
                        split_mask, clone_mask = new_split, new_clone
                    
                    n_clone = clone_mask.sum().item()
                    n_split = split_mask.sum().item()
                    if n_clone > 0 or n_split > 0:
                        print(f"  Densification (Accumulated): Clones: {n_clone}, Splits: {n_split}")
                        self.gaussians.densify_clone_split(clone_mask, split_mask)
                
                # Clear accumulation buffers after densification
                self.xyz_gradient_accum.fill_(0)
                self.denom.fill_(0)
                self.max_radii2D.fill_(0)

            # Pruning: opacity + size based
            with torch.no_grad():
                current_count = self.gaussians.xyz.shape[0]
                if current_count > self.min_points:
                    # Prune nearly-invisible Gaussians
                    opacity_mask = (self.gaussians.opacity.squeeze() < 0.005)
                    # Prune Gaussians that are way too large (blobs)
                    max_scales = torch.max(self.gaussians.scales, dim=1).values
                    size_mask = (max_scales > 0.5)

                    prune_mask = opacity_mask | size_mask
                    
                    n_prune = prune_mask.sum().item()
                    if n_prune > 0 and (current_count - n_prune) >= self.min_points:
                        print(f"  Pruning {n_prune} points (Opacity: {opacity_mask.sum().item()}, Size: {size_mask.sum().item()})")
                        self.gaussians.prune_points(prune_mask)
        
        # Final scalar safety check
        loss_val = loss.item() if loss.dim() == 0 else loss.mean().item()
        return loss_val, rendered_image
