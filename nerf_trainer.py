import torch
import torch.optim as optim
from torch.utils.data import DataLoader
class NeRFTrainer:
    """
    Manages Optimization loop to learn geometry from COLMAP camera poses, keeping GPU memory limits Safe"""
    def __init__(self,model,learning_rate=5e-4):
        self.model=model
        self.optimizer=optim.Adam(self.model.parameters(),lr=learning_rate)
        self.chunk_size=4096#adjust if gpu runs out of memory
    def render_rays_in_chunks(self,rays_o,rays_d):
        '''
        Processes rays in smaller chunks to prevent CUDA out of memory errors when rendering high resolution room views'''
        all_rgb=[]
        all_density=[]

        rays_o_flat=rays_o.reshape(-1,3)
        rays_d_flat=rays_d.reshape(-1,3)
        for i in range(0,rays_o_flat.shape[0],self.chunk_size):
            chunk_o=rays_o_flat[i:i+self.chunk_size]
            chunk_d=rays_d_flat[i:i+self.chunk_size]

            sampled_points=chunk_o.unsqueeze(1)+chunk_d.unsqueeze(1)*torch.linspace(0.1,10.0,64).cuda()
            predictions=self.model(sampled_points)
            all_rgb.append(predictions[...,:3])
            all_density.append(predictions[...,3])
        return torch.cat(all_rgb,dim=0),torch.cat(all_density,dim=0)
    def train_step(self,image_batch,pose_batch):
        '''Executes a single training iteration'''
        self.model.train()
        self.optimizer.zero_grad()
        H,W,K,c2w=pose_batch
        rays_o,rays_d=self._get_rays(H,W,K,c2w)#generates rays based on camera poses,
        pred_rgb,pred_density=self.render_rays_in_chunks(rays_o,rays_d)#render the chunked rays
        pred_rgb=pred_rgb.view(H,W,3)
        loss=torch.nn.functional.mse_loss(pred_rgb,image_batch)#compute loss
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
        dirs=torch.stack([(i-cx)/fx,-(j-cy)/fy,-torch.ones_like(i,device='cuda')],dim=-1)
        rotation_matrix=c2w[:3,:3]
        rays_d=torch.sum(dirs[...,None,:]*rotation_matrix,dim=-1)
        rays_d=rays_d/torch.norm(rays_d,dim=-1,keepdim=True)

        rays_o=c2w[:3,3].expand(rays_d.shape)

        return rays_o,rays_d
