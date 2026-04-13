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
    def generate_initial_gaussians(self,resolution=256,threshold=0.5):
        '''
        Samples room's volume in chunks to find solid surfaces
        '''
        print(f"Generating grid at {resolution}^3 resolution")
        grid_coords=self._create_room_grid(resolution).cuda()
        
        all_densities = []
        all_rgbs = []
        
        for i in range(0,grid_coords.shape[0],self.chunk_size):
            chunk=grid_coords[i:i+self.chunk_size]
            with torch.no_grad():
                predictions=self.nerf(chunk)
                # RGB is already sigmoid-clamped by the model
                rgb=predictions[...,:3]
                # Density is the 4th channel
                density=torch.relu(predictions[...,3])
                
                all_densities.append(density)
                all_rgbs.append(rgb)
                
        full_density = torch.cat(all_densities, dim=0)
        full_rgb = torch.cat(all_rgbs, dim=0)
        
        mask = full_density > threshold
        if not mask.any():
            print(f"Warning: No points found at threshold {threshold}.")
            max_dens = full_density.max().item()
            print(f"Maximum density in volume is {max_dens:.4f}")
            if max_dens <= 0:
                raise ValueError("NeRF model outputting zero density everywhere. Training likely failed.")
            
            # Dynamically set threshold to a fraction of max density to ensure extraction
            threshold = max(0.1, max_dens * 0.5)
            print(f"Adjusting threshold to {threshold:.4f} and trying again...")
            mask = full_density > threshold
            
        final_xyz = grid_coords[mask]
        final_colors = full_rgb[mask]
        
        # Limit to 200k points max to prevent OOM during co-optimization
        if final_xyz.shape[0] > 200000:
             print(f"Cap reached: Subsampling {final_xyz.shape[0]} down to 200k points.")
             indices = torch.randperm(final_xyz.shape[0])[:200000]
             final_xyz = final_xyz[indices]
             final_colors = final_colors[indices]

        if final_xyz.shape[0] == 0:
             raise ValueError("No dense points found even after threshold adjustment.")
             
        print(f"Bridge Completed:Extracted {final_xyz.shape[0]} Gaussian primitives")
        return final_xyz,final_colors
    def _create_room_grid(self,res):
        '''Generates 3D Coordinates spanning the bounding box'''
        x=torch.linspace(self.bbox[0][0],self.bbox[1][0],res)
        y=torch.linspace(self.bbox[0][1],self.bbox[1][1],res)
        z=torch.linspace(self.bbox[0][2],self.bbox[1][2],res)
        X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
        return torch.stack([X,Y,Z],dim=-1).reshape(-1,3)
    

        