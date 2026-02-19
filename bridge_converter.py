import torch
import numpy as np
class NeRFToGaussianBridge:
    '''
    Implements the hybrid transition by sampling NeRF density field to initialize Gaussian primitives
    '''
    def __init__(self,nerf_model,bounding_box):
        self.nerf=nerf_model
        self.bbox=bounding_box
    def generate_initial_gaussians(self,resolution=128,threshold=15.0):
        '''
        Samples NeRF field for high density points to server as Gaussian means
        '''
        grid=self._create_sample_grid(resolution)
        with torch.no_grad():
            densities=self.nerf.query_density(grid)
        mask=densities>threshold
        xyz_inital=grid[mask]
        colors=self.nerf.query_color(xyz_inital)
        return xyz_inital,colors
    def _create_sample_grid(self,res):
        x=torch.linspace(self.bbox[0],self.bbox[1],res)
        y=torch.linspace(self.bbox[2],self.bbox[3],res)
        z=torch.linspace(self.bbox[4],self.bbox[5],res)
        return torch.stack(torch.meshgrid(x,y,z,indexing='ij'),dim=1).reshape(-1,3)
    
