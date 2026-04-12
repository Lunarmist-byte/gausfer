import torch
import torch.optim as optim
from gaussian_model import GaussianModel
from HybridCoOptimizer import HybridCoOptimizer
from main import update_optimizer

class DummyNeRF:
    def __init__(self):
        pass
    def __call__(self, xyz):
        # returns density > 0.1 for all
        return torch.ones((xyz.shape[0], 4), device='cuda') * 25.0

def main():
    gaussians = GaussianModel(sh_degree=1)
    # Init 100 points
    xyz = torch.rand((100, 3), device='cuda')
    rgb = torch.rand((100, 3), device='cuda')
    gaussians.initialize_from_nerf(xyz, rgb)
    
    gaussians.xyz.retain_grad()
    gaussians.scales.retain_grad()
    
    # Simulate some gradients
    optimizer = optim.Adam(gaussians.parameters(), lr=0.01)
    
    # We need a grad to trigger densification
    loss = gaussians.xyz.sum() + gaussians.scales.sum()
    loss.backward()
    
    # artificially inflate grads to pass threshold
    gaussians.xyz.grad = torch.ones_like(gaussians.xyz) * 1.0
    
    # artificially set scales
    gaussians._scaling.data[:50] = torch.log(torch.tensor(0.2, device='cuda'))
    gaussians._scaling.data[50:] = torch.log(torch.tensor(0.01, device='cuda'))
    
    nerf = DummyNeRF()
    co_optimizer = HybridCoOptimizer(nerf, gaussians, grad_threshold=0.5, density_threshold=10.0)
    
    old_count = gaussians.xyz.shape[0]
    print(f"Old count: {old_count}")
    
    with torch.no_grad():
        grads = gaussians.xyz.grad
        high_error_mask = torch.norm(grads,dim=-1) > co_optimizer.grad_threshold
        problematic_xyz = gaussians.xyz[high_error_mask]
        nerf_density = torch.relu(nerf(problematic_xyz)[...,3])
        valid_split_mask = nerf_density > co_optimizer.density_threshold
        
        combined_mask = torch.zeros_like(high_error_mask)
        combined_mask[high_error_mask] = valid_split_mask
        
        max_scales = torch.max(gaussians.scales, dim=1).values
        
        split_mask = combined_mask & (max_scales > 0.1)
        clone_mask = combined_mask & (max_scales <= 0.1)
        
        print(f"Split mask count: {split_mask.sum()}")
        print(f"Clone mask count: {clone_mask.sum()}")
        
        gaussians.densify_clone_split(clone_mask, split_mask)
        
    new_count = gaussians.xyz.shape[0]
    print(f"New count: {new_count}")
    
    optimizer = update_optimizer(optimizer, gaussians.parameters(), 0.01)
    print("Optimizer updated successfully")
    
if __name__ == "__main__":
    main()
