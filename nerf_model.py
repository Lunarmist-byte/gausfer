import torch
import torch.nn as nn
import torch.nn.functional as F
class NeRFNetwork(nn.Module):
    '''FR3:A Volumetric Representation that predicts color and density'''
    def __init__(self,d_input=3,d_filter=256):
        super(NeRFNetwork,self).__init__()
        self.layers=nn.ModuleList([nn.Linear(d_input,d_filter),nn.Linear(d_filter,d_input),nn.Linear(d_filter,d_filter+1)])
    def forward(self,x):
        '''Processes 3D coordinates to output scene density and RGB'''
        h=x
        for i,l in enumerate(self.layers):
            h=F.relu(l(h))
        sigma=F.relu(h[...,0])
        rgb=torch.sigmoid(h[...,1:])
        return rgb,sigma
    def query_density(self,x):
        _,sigma=self.forward(x)
        return sigma