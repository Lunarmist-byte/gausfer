import torch
import torch.optim as optim
from torch.utils.data import DataLoader
class NeRFTrainer:
    """
    Manages Optimization loop to learn geometry from COLMAP camera poses, keeping GPU memory limits Safe"""
    def __init__(self,model,learning_rate=5e-4, near=0.1, far=10.0):
        self.model=model
        self.optimizer=optim.Adam(self.model.parameters(),lr=learning_rate)
        self.chunk_size=1024#reduced for OOM
        self.near = near
        self.far = far
        self.num_samples = 64
    def render_rays_in_chunks(self,rays_o,rays_d):
        '''
        Processes rays in smaller chunks to prevent CUDA out of memory errors when rendering high resolution room views'''
        all_rgb=[]
        
        rays_o_flat=rays_o.reshape(-1,3)
        rays_d_flat=rays_d.reshape(-1,3)
        
        # Set up sampling distances
        t_vals = torch.linspace(self.near, self.far, self.num_samples).cuda()
        dists = t_vals[1:] - t_vals[:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device='cuda')], dim=0) # last dist is infinity
        
        for i in range(0,rays_o_flat.shape[0],self.chunk_size):
            chunk_o=rays_o_flat[i:i+self.chunk_size]
            chunk_d=rays_d_flat[i:i+self.chunk_size]

            sampled_points=chunk_o.unsqueeze(1)+chunk_d.unsqueeze(1)*t_vals[None,:,None]
            predictions=self.model(sampled_points) # (C, S, 4)
            
            rgb = predictions[..., :3] # (C, S, 3)
            density = torch.relu(predictions[..., 3]) # (C, S)
            
            # Simple volume rendering (Alpha-blending)
            alpha = 1.0 - torch.exp(-density * dists[None, :]) # (C, S)
            weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device='cuda'), 1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
            
            rendered_rgb = torch.sum(weights.unsqueeze(-1) * rgb, dim=1) # (C, 3)
            all_rgb.append(rendered_rgb)
            
        return torch.cat(all_rgb,dim=0), None # Return None for density as it's not used by main loop yet

    def train_step(self,image_batch,pose_batch):
        '''Executes a single training iteration using random ray sampling'''
        self.model.train()
        self.optimizer.zero_grad()
        H,W,K,c2w=pose_batch
        
        # Flatten image to sample rays
        img_flat = image_batch.reshape(-1, 3)
        rays_o, rays_d = self._get_rays(H, W, K, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        # Sample 4096 rays for better learning
        num_rays = 4096
        coords = torch.randint(0, H*W, (num_rays,), device='cuda')
        
        select_o = rays_o[coords]
        select_d = rays_d[coords]
        target_rgb = img_flat[coords]
        
        # Render sampled rays (manually for training to avoid all_rgb accumulation)
        t_vals = torch.linspace(self.near, self.far, self.num_samples).cuda()
        dists = t_vals[1:] - t_vals[:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device='cuda')], dim=0)
        
        sampled_points = select_o.unsqueeze(1) + select_d.unsqueeze(1) * t_vals[None, :, None]
        predictions = self.model(sampled_points)
        
        rgb = predictions[..., :3]
        density = torch.relu(predictions[..., 3])
        
        alpha = 1.0 - torch.exp(-density * dists[None, :])
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device='cuda'), 1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        
        pred_rgb = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
        
        loss=torch.nn.functional.mse_loss(pred_rgb,target_rgb)#compute loss
        loss.backward()#update weights
        self.optimizer.step()

        return loss.item()
    def _get_rays(self,H,W,K,c2w):
        '''calculates how physical light rays passing through the camera lens in the room
            Vectorized to run parallel on CUDA without slow loops
        '''
        i,j=torch.meshgrid(torch.linspace(0,W-1,W,device='cuda'),torch.linspace(0,H-1,H,device='cuda'),indexing='ij')
        i=i.t()
        j=j.t()
        fx,fy=K[0,0],K[1,1]
        cx,cy=K[0,2],K[1,2]
        dirs=torch.stack([(i-cx)/fx,(j-cy)/fy,torch.ones_like(i,device='cuda')],dim=-1)
        rotation_matrix=c2w[:3,:3]
        rays_d=torch.sum(dirs[...,None,:]*rotation_matrix,dim=-1)
        rays_d=rays_d/torch.norm(rays_d,dim=-1,keepdim=True)

        rays_o=c2w[:3,3].expand(rays_d.shape)

        return rays_o,rays_d
