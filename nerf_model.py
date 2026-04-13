import torch
import torch.nn as nn
class PositionalEncoding(nn.Module):
    '''Maps 3D Coordinates to higher frequency space so MLP can learn sharp corners'''
    def __init__(self,num_frequencies):
        super().__init__()
        self.num_frequencies=num_frequencies    
    def forward(self,x):
        '''Processes 3D coordinates to output scene density and RGB'''
        freqs=[x]
        for i in range(self.num_frequencies):
            freqs.append(torch.sin((2**i)*x))
            freqs.append(torch.cos((2**i)*x))
        return torch.cat(freqs,dim=-1)
class RoomNeRF(nn.Module):
    """
    FR3:Volumetric scene Representation optimized for CUDA execution
    Designed to process encoded 3D spatial coordinates and viewing directions
    Uses a skip connection at the midpoint for better geometry learning.
    """
    def __init__(self,num_freqs=10,hidden_dim=256):
        super().__init__()
        in_dim=3+2*3*num_freqs
        self.encoder=PositionalEncoding(num_freqs)
        # First half of the network
        self.block1=nn.Sequential(
            nn.Linear(in_dim,hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
        ).cuda()
        # Second half with skip connection (input encoding concatenated back in)
        self.block2=nn.Sequential(
            nn.Linear(hidden_dim+in_dim,hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
        ).cuda()
        # Separate heads for density and RGB
        self.density_head=nn.Linear(hidden_dim,1).cuda()
        self.rgb_head=nn.Linear(hidden_dim,3).cuda()
    def forward(self,x):
        encoded_x=self.encoder(x)
        h=self.block1(encoded_x)
        # Skip connection: concatenate original encoding
        h=self.block2(torch.cat([h,encoded_x],dim=-1))
        density=self.density_head(h)
        rgb=torch.sigmoid(self.rgb_head(h))#Clamped to [0,1]
        return torch.cat([rgb,density],dim=-1)
    