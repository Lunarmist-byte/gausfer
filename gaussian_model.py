import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter

class GaussianModel(nn.Module):
    """
    Stores and manages explicit 3D Gaussian Primitives.
    Handles the mathematical conversions
    """
    def __init__(self,sh_degree=3):
        super().__init__()
        self.max_sh_degree=sh_degree
        self.active_sh_degree=0 # Start at degree 0 for stability
        #core parameters that the Co-Optimizer will update
        self._xyz=torch.empty(0)
        self._features_dc=torch.empty(0)#Diffuse to Base RGB
        self._features_rest=torch.empty(0)#View-dependent effects
        self._scaling=torch.empty(0)
        self._rotation=torch.empty(0)
        self._opacity=torch.empty(0)
    def initialize_from_pcd(self, xyz, rgb, opacities=None):
        '''Initializes Gaussians directly from SfM point cloud for maximum reliability'''
        # Hard cap: diff_gaussian_rasterization tile buffer cannot handle >50k splats safely
        MAX_INIT_POINTS = 50000
        if xyz.shape[0] > MAX_INIT_POINTS:
            indices = torch.randperm(xyz.shape[0], device=xyz.device)[:MAX_INIT_POINTS]
            xyz = xyz[indices]
            rgb = rgb[indices]
            if opacities is not None:
                opacities = opacities[indices]
            print(f"  Downsampled SfM seed from {xyz.shape[0]+MAX_INIT_POINTS} -> {MAX_INIT_POINTS} points to fit CUDA tile buffer.")
        print(f"Initializing {xyz.shape[0]} Gaussians from SfM Point Cloud...")
        self._initialize_params(xyz, rgb, opacities)

    def initialize_from_nerf(self, init_xyz, init_rgb):
        '''Takes dense point cloud extracted by bridge_converter.py and initializes the mathematical properties of Gaussians'''
        MAX_INIT_POINTS = 50000
        if init_xyz.shape[0] > MAX_INIT_POINTS:
            indices = torch.randperm(init_xyz.shape[0], device=init_xyz.device)[:MAX_INIT_POINTS]
            init_xyz = init_xyz[indices]
            init_rgb = init_rgb[indices]
            print(f"  Downsampled NeRF seed to {MAX_INIT_POINTS} points.")
        print(f"Initializing {init_xyz.shape[0]} Gaussians from NeRF Seed...")
        self._initialize_params(init_xyz, init_rgb)

    def _initialize_params(self, xyz, rgb, opacities=None):
        # Spherical Harmonics (Color)
        C0 = 0.28209479177387814
        f_dc = (rgb - 0.5) / C0

        # Robust Scaling: Use distance to nearest neighbors
        scales = self._compute_initial_scaling(xyz)

        # Rotation
        rots = torch.zeros((xyz.shape[0], 4), device='cuda')
        rots[:, 0] = 1.0

        # Opacity
        if opacities is None:
            opacities = torch.logit(torch.ones((xyz.shape[0], 1), device='cuda') * 0.5)

        self._xyz = Parameter(xyz.requires_grad_(True))
        self._features_dc = Parameter(f_dc.unsqueeze(1).requires_grad_(True))
        
        # Allocate rest SH only for active degree (0 at start = zero extra VRAM)
        # Will be expanded when increase_sh_degree() is called
        n_rest = (self.max_sh_degree + 1) ** 2 - 1
        features_rest = torch.zeros((xyz.shape[0], n_rest, 3), device='cuda')
        self._features_rest = Parameter(features_rest.requires_grad_(True))
        
        self._scaling = Parameter(scales.requires_grad_(True))
        self._rotation = Parameter(rots.requires_grad_(True))
        self._opacity = Parameter(opacities.requires_grad_(True))

    def _compute_initial_scaling(self, xyz):
        """Calculates distance to 3-nearest neighbors to set initial Gaussian sizes"""
        print("  Calculating initial scaling factors...")
        # For large point clouds, we sample a subset to keep it fast
        if xyz.shape[0] > 10000:
            indices = torch.randperm(xyz.shape[0])[:10000]
            sample_xyz = xyz[indices]
        else:
            sample_xyz = xyz

        # Distance matrix for the sampled points
        dist_mat = torch.cdist(sample_xyz, sample_xyz)
        # Find 4 nearest (including self)
        nearest_dists = torch.topk(dist_mat, k=4, largest=False, dim=-1).values[:, 1:] # Skip self
        avg_dist = nearest_dists.mean(dim=-1).mean()
        
        # Log-scale for GaussianModel
        # Multiply by 2.0 (down from 3.0) for tighter initial fit
        initial_scale = torch.clamp(avg_dist * 2.0, min=0.001, max=0.05)
        print(f"  Determined average point distance: {initial_scale:.4f}")
        return torch.log(torch.full((xyz.shape[0], 3), initial_scale, device='cuda'))

    #Property Getters for the Rasterizer
    @property
    def xyz(self):
        return self._xyz.contiguous()
    @property
    def shs(self):
        # Mask out SH degrees beyond the active one to allow gradual warmup
        num_sh_coeffs = (self.active_sh_degree + 1) ** 2
        shs_full = torch.cat([self._features_dc, self._features_rest], dim=1)
        return shs_full[:, :num_sh_coeffs, :].contiguous()
    @property
    def opacity(self):
        return torch.sigmoid(self._opacity).contiguous()
    @property
    def scales(self):
        # Clamp minimum scale to prevent Gaussians from collapsing to zero
        return torch.clamp(torch.exp(self._scaling), min=1e-6).contiguous()
    @property
    def rotations(self):
        return torch.nn.functional.normalize(self._rotation).contiguous()
    
    def increase_sh_degree(self):
        '''Gradually increase active SH degree for stable color warmup'''
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print(f"  SH degree increased to {self.active_sh_degree}")
    
    def densify_clone_split(self, clone_mask, split_mask):
        new_xyz = []
        new_features_dc = []
        new_features_rest = []
        new_scaling = []
        new_rotation = []
        new_opacity = []

        if clone_mask is not None and clone_mask.any():
            new_xyz.append(self._xyz[clone_mask])
            new_features_dc.append(self._features_dc[clone_mask])
            new_features_rest.append(self._features_rest[clone_mask])
            new_scaling.append(self._scaling[clone_mask])
            new_rotation.append(self._rotation[clone_mask])
            new_opacity.append(self._opacity[clone_mask])

        if split_mask is not None and split_mask.any():
            split_xyz = self._xyz[split_mask]
            split_features_dc = self._features_dc[split_mask]
            split_features_rest = self._features_rest[split_mask]
            split_rotation = self._rotation[split_mask]
            split_opacity = self._opacity[split_mask]
            
            # Shrink existing splats before splitting
            self._scaling.data[split_mask] -= torch.log(torch.tensor(1.6, device='cuda'))
            # Hard-clamp in log-space: exp(-6)=0.002, exp(1)=2.7 max world units
            self._scaling.data.clamp_(-6.0, 1.0)
            split_scaling = self._scaling[split_mask]
            
            # Add displacement for new points - clamp std to prevent explosion
            std = torch.exp(split_scaling).clamp(max=0.5)
            noise = torch.randn_like(split_xyz) * std
            split_xyz_new = split_xyz + noise
            # Guard: replace any NaN/Inf positions with the original xyz
            bad_mask = ~torch.isfinite(split_xyz_new).all(dim=-1)
            split_xyz_new[bad_mask] = split_xyz[bad_mask]

            new_xyz.append(split_xyz_new)
            new_features_dc.append(split_features_dc)
            new_features_rest.append(split_features_rest)
            new_scaling.append(split_scaling)
            new_rotation.append(split_rotation)
            # Reduce opacity of split points slightly (0.8x) to ensure overlap doesn't cause saturation
            new_opacity.append(split_opacity - 0.223) # logit domain approx factor

        if new_xyz:
            self._xyz = Parameter(torch.cat([self._xyz] + new_xyz, dim=0).requires_grad_(True))
            self._features_dc = Parameter(torch.cat([self._features_dc] + new_features_dc, dim=0).requires_grad_(True))
            self._features_rest = Parameter(torch.cat([self._features_rest] + new_features_rest, dim=0).requires_grad_(True))
            self._scaling = Parameter(torch.cat([self._scaling] + new_scaling, dim=0).requires_grad_(True))
            self._rotation = Parameter(torch.cat([self._rotation] + new_rotation, dim=0).requires_grad_(True))
            self._opacity = Parameter(torch.cat([self._opacity] + new_opacity, dim=0).requires_grad_(True))
            return True
        return False

    def prune_points(self,prune_mask):
        #Delete gaussians that are transparents
        keep_mask=~prune_mask

        self._xyz=Parameter(self._xyz[keep_mask].requires_grad_(True))
        self._features_dc=Parameter(self._features_dc[keep_mask].requires_grad_(True))
        self._features_rest=Parameter(self._features_rest[keep_mask].requires_grad_(True))
        self._scaling=Parameter(self._scaling[keep_mask].requires_grad_(True))
        self._rotation=Parameter(self._rotation[keep_mask].requires_grad_(True))
        self._opacity=Parameter(self._opacity[keep_mask].requires_grad_(True))

        return True

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        """Saves current state of the Gaussians to a standard .ply file compatible with 3DGS viewers"""
        print(f"Exporting 3D Gaussians to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1).astype(np.float32)

        with open(path, 'wb') as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n".encode('utf-8'))
            
            for attr in self.construct_list_of_attributes():
                f.write(f"property float {attr}\n".encode('utf-8'))
                
            f.write(b"end_header\n")
            f.write(attributes.tobytes())

    def save_checkpoint(self, path):
        """Saves PyTorch state for seamless resuming"""
        print(f"Saving Gaussian Checkpoint to {path}...")
        torch.save({
            "_xyz": self._xyz.data,
            "_features_dc": self._features_dc.data,
            "_features_rest": self._features_rest.data,
            "_scaling": self._scaling.data,
            "_rotation": self._rotation.data,
            "_opacity": self._opacity.data,
        }, path)

    def load_checkpoint(self, path):
        """Restores PyTorch state, dynamically re-sizing Parameters with Self-Healing"""
        print(f"Loading Gaussian Checkpoint from {path}...")
        ckpt = torch.load(path)
        
        # Self-Healing: Check for NaNs/Infs
        xyz = ckpt["_xyz"]
        nan_mask = torch.isnan(xyz).any(dim=-1)
        inf_mask = torch.isinf(xyz).any(dim=-1)
        invalid_mask = nan_mask | inf_mask
        
        if invalid_mask.any():
            print(f"  [Safety] Found {invalid_mask.sum().item()} points with NaNs/Infs. Scrubbing...")
            clean_mask = ~invalid_mask
            for key in ckpt.keys():
                ckpt[key] = ckpt[key][clean_mask]
        
        # Self-Healing: Cap oversized points that cause rasterizer crashes
        scales = ckpt["_scaling"]
        oversized_mask = (torch.max(scales, dim=-1).values > 0.5) # World space log(exp(0.5))
        if oversized_mask.any():
            print(f"  [Safety] Flattening {oversized_mask.sum().item()} oversized Gaussians.")
            ckpt["_scaling"][oversized_mask] = -1.0 # Cap to approx exp(-1) = 0.36
            
        # VRAM Safety: If still too many points for stable resume, subsample
        max_resume_points = 1000000
        if ckpt["_xyz"].shape[0] > max_resume_points:
            print(f"  [Safety] Checkpoint has {ckpt['_xyz'].shape[0]} points. Downsampling to {max_resume_points} for VRAM stability...")
            indices = torch.randperm(ckpt["_xyz"].shape[0])[:max_resume_points]
            for key in ckpt.keys():
                ckpt[key] = ckpt[key][indices]

        self._xyz = Parameter(ckpt["_xyz"].cuda().requires_grad_(True))
        self._features_dc = Parameter(ckpt["_features_dc"].cuda().requires_grad_(True))
        self._features_rest = Parameter(ckpt["_features_rest"].cuda().requires_grad_(True))
        self._scaling = Parameter(ckpt["_scaling"].cuda().requires_grad_(True))
        self._rotation = Parameter(ckpt["_rotation"].cuda().requires_grad_(True))
        self._opacity = Parameter(ckpt["_opacity"].cuda().requires_grad_(True))

