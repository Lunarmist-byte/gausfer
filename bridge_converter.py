import torch
import numpy as np
class NeRFToGaussianBridge:
    '''
    Implements the hybrid transition by sampling NeRF density field to initialize Gaussian primitives
    Extracts high-density geometry from trained NeRF Model
    3DGS primitives for a room
    '''
    def __init__(self,nerf_model,bounding_box):
        self.nerf=nerf_model
        self.bbox=bounding_box
        self.chunk_size=1024*64#edit if needed, to prevent memory overflow for gpu
    def generate_initial_gaussians(self,resolution=256,threshold=15.0):
        '''
        Samples room's volume in chunks to find solid surfaces
        '''
        print(f"Generating grid at {resolution}^3 resolution")
        grid_coords=self._create_room_grid(resolution).cuda()
        extracted_xyz=[]
        extracted_colors=[]
        for i in range(0,grid_coords.shape[0],self.chunk_size):
            chunk=grid_coords[i:i+self.chunk_size]
            with torch.no_grad():
                predictions=self.nerf(chunk)
                density=predictions[...,3]
                rgb=predictions[...,:3]
            mask=density>threshold
            if mask.any():
                extracted_xyz.append(chunk[mask])
                extracted_colors.append(rgb[mask])
        if not extracted_xyz:
            raise ValueError("No dense points found,Check NeRF tarining or lower threshold")
        final_xyz=torch.cat(extracted_xyz,dim=0)
        final_colors=torch.cat(extracted_colors,dim=0)
        print(f"Bridge Completed:Extracted{final_xyz.shape[0]} Gaussian primitives")
        return final_xyz,final_colors
    def _create_room_grid(self,res):
        '''Generates 3D Coordinates spanning the bounding box'''
        x=torch.linspace(self.bbox[0][0],self.bbox[1][0],res)
        y=torch.linspace(self.bbox[0][1],self.bbox[1][1],res)
        z=torch.linspace(self.bbox[0][2],self.bbox[1][2],res)
        X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
        return torch.stack([X,Y,Z],dim=1).reshape(-1,3)
    

        