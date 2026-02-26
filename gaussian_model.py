import torch
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
        #core parameters that the Co-Optimizer will update
        self._xyz=torch.empty(0)
        self._features_dc=torch.empty(0)#Diffuse to Base RGB
        self._scaling=torch.empty(0)
        self._rotation=torch.empty(0)
        self._opacity=torch.empty(0)
    def initialize_from_nerf(self,init_xyz,init_rgb):
        '''
        Takes dense point cloud extracted by bridge_converter.py and 
        initializes the mathematical properties of Gaussians
        '''
        print(f"Initializing{init_xyz.shape[0]}Gaussians from NeRF Seed...")
        #Spherical Harmonics(Color)
        #standard rgb to Spherical Harmonic DC component
        C0=0.28209479177387814
        f_dc=(init_rgb-0.5)/C0
        #Scaling
        inital_scale=torch.ones_like(init_xyz[:,0])*0.01
        scales=torch.log(inital_scale)[...,None].repeat(1,3)
        #Rotation
        rots=torch.zeros((init_xyz.shape[0],4),device='cuda')
        rots[:,0]=1.0
        #Opacity
        opacities=torch.logit(torch.ones((init_xyz.shape[0],1),device='cuda')*0.1)
        self._xyz=Parameter(init_xyz.requires_grad_(True))
        self._features_dc=Parameter(f_dc.unsqueeze(1).requires_grad_(True))
        self._scaling=Parameter(scales.requires_grad_(True))
        self._rotation=Parameter(rots.requires_grad_(True))
        self._opacity=Parameter(opacities.requires_grad_(True))

    #Property Getters for the Rasterizer
    @property
    def xyz(self):
        return self._xyz
    @property
    def shs(self):
        return self._features_dc
    @property
    def opacity(self):
        return torch.sigmoid(self._opacity)
    @property
    def scales(self):
        return torch.exp(self._scaling)
    @property
    def rotations(self):
        return torch.nn.functional.normalize(self._rotation)
    
    #Topology Management for Co-Optimizer
    def densify_and_split(self,split_mask):
        #Extract Properties
        new_xyz=self._xyz[split_mask]
        new_features_dc=self._features_dc[split_mask]
        new_rotation=self._rotation[split_mask]
        new_opacity=self._opacity[split_mask]
        #Shrink
        new_scaling=self._scaling[split_mask]-torch.log(torch.tensor(1.6,device='cuda'))
        self._scaling.data[split_mask]=new_scaling
        #Concatenate
        self._xyz=Parameter(torch.cat([self._xyz,new_xyz],dim=0).requires_grad_(True))
        self._features_dc=Parameter(torch.cat([self._features_dc,new_features_dc],dim=0).requires_grad_(True))
        self._scaling=Parameter(torch.cat([self._scaling,new_scaling],dim=0).requires_grad_(True))
        self._rotation=Parameter(torch.cat([self._rotation,new_rotation],dim=0).requires_grad_(True))
        self._opacity=Parameter(torch.cat([self._opacity],new_opacity),dim=0).requires_grad_(True)
        return True

    def prune_points(self,prune_mask):
        #Delete gaussians that are transparents
        keep_mask=~prune_mask

        self._xyz=Parameter(self._xyz[keep_mask].requires_grad_(True))
        self._features_dc=Parameter(self._features_dc[keep_mask]),self.requires_grad_(True)
        self._scaling=Parameter(self._scaling[keep_mask].requires_grad_(True))
        self._rotation=Parameter(self._rotation[keep_mask].requires_grad_(True))
        self._opacity=Parameter(self._opacity[keep_mask].requires_grad_(True))

        return True


