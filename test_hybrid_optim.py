import torch
import torch.optim as optim
from gaussian_model import GaussianModel
from HybridCoOptimizer import HybridCoOptimizer
from main import update_optimizer

class DummyNeRF(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, xyz):
        # returns density > 0.1 for all
        return torch.ones((xyz.shape[0], 4), device='cuda') * 25.0

def main():
    torch.manual_seed(42)
    gaussians = GaussianModel(sh_degree=1)
    
    # Init 100 points
    xyz = torch.rand((100, 3), device='cuda')
    rgb = torch.rand((100, 3), device='cuda')
    gaussians.initialize_from_nerf(xyz, rgb)
    
    # Verify constraints on init
    gaussians.constrain_parameters()
    print("Initial constraints applied.")

    nerf = DummyNeRF()
    co_optimizer = HybridCoOptimizer(nerf, gaussians, grad_threshold=0.5)
    
    # Simulate a step to populate gradients
    print(f"Old count: {gaussians.xyz.shape[0]}")
    
    # Manually trigger densification logic to test safety
    # In a real run, this happens inside co_optimizer.step()
    # We'll simulate the state co_optimizer.step() would produce
    co_optimizer.xyz_gradient_accum.fill_(1.0) # High error
    co_optimizer.denom.fill_(1.0)
    
    # Set some large scales to test splitting
    with torch.no_grad():
        gaussians._scaling.data[:10] = 0.4 # Will be split
        gaussians._scaling.data[10:20] = -5.0 # Will be cloned
        
    # Trigger densification
    clone_mask = torch.zeros(gaussians.xyz.shape[0], dtype=torch.bool, device='cuda')
    split_mask = torch.zeros(gaussians.xyz.shape[0], dtype=torch.bool, device='cuda')
    
    # Manually pick points
    clone_mask[10:20] = True
    split_mask[:10] = True
    
    print(f"Triggering densification: {clone_mask.sum().item()} clones, {split_mask.sum().item()} splits")
    gaussians.densify_clone_split(clone_mask, split_mask)
    
    print(f"New count: {gaussians.xyz.shape[0]}")
    
    # Verify constraints after densification
    for name, param in [("xyz", gaussians._xyz), ("scaling", gaussians._scaling), ("rotation", gaussians._rotation)]:
        if not torch.isfinite(param.data).all():
            print(f"ERROR: {name} contains non-finite values!")
        else:
            print(f"SUCCESS: {name} is finite.")
            
    print("Scaling range:", gaussians._scaling.data.min().item(), "to", gaussians._scaling.data.max().item())
    
    # Verify optimizer update logic
    param_groups = [
        {"params": [gaussians._xyz], "lr": 0.001, "name": "xyz"},
        {"params": [gaussians._features_dc], "lr": 0.0025, "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": 0.000125, "name": "f_rest"},
        {"params": [gaussians._opacity], "lr": 0.05, "name": "opacity"},
        {"params": [gaussians._scaling], "lr": 0.005, "name": "scaling"},
        {"params": [gaussians._rotation], "lr": 0.001, "name": "rotation"},
    ]
    dummy_optim = optim.Adam(param_groups, lr=0.0)
    new_optim = update_optimizer(dummy_optim, param_groups)
    print("Optimizer updated successfully.")

if __name__ == "__main__":
    main()

