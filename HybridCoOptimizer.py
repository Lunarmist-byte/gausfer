import torch
class HybridCoOptimizer:
    '''implements nerf directly to gaussian
    uses volumetric understanding of NeRF to guide the densification
    and pruning of 3DGS'''
    def __init__(self,nerf_model,gaussian_model):
        self.nerf=nerf_model
        self.gaussians=gaussian_model
        self.grad_threshold=0.0002
    def step(self,view_cam,ground_truth_image,render_func):
        #render the gaussians
        render_pkg=render_func(view_cam,self.gaussians)
        rendered_image=render_pkg["render"]
        #calculate rasterization loss
        loss=torch.nn.functional.l1_loss(rendered_image,ground_truth_image)
        loss.backward()
        #Nerf Guided Densification
        with torch.no_grad():
            grads=self.gaussians.xyz.grad
            high_error_mask=torch.norm(grads,dim=-1)>self.grad_threshold
            if high_error_mask.any():
                problematic_xyz=self.gaussians.xyz[high_error_mask]
                #asking if nerf detects solid matter
                nerf_density=self.nerf(problematic_xyz)[...,3]
                valid_split_mask=nerf_density>5.0

                if valid_split_mask.any():
                    combined_mask=torch.zeros_like(high_error_mask)
                    combined_mask[high_error_mask]=valid_split_mask
                    
                    self.gaussians.densify_and_split(combined_mask)
        #nerf-guided pruning
        with torch.no_grad():
            current_density=self.nerf(self.gaussians.xyz)[...,3]
            prune_mask=(self.gaussians.opacity.squeeze()<0.05)&(current_density<1.0)
            if prune_mask.any():
                self.gaussians.prune_points(prune_mask)

