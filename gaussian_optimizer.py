import torch
class GaussianOptimizer:
    '''Handles scaling the reconstruction to full rooms by densifying and pruning Gaussians'''
    def __init__(self,gaussain_model,config):
        self.model=gaussain_model
        self.densify_grad_threshold=0.0002
        self.opacity.threshold=0.01
    def densify_and_prune(self,iteration,grads):
        '''Splits the gaussians into high gradients and removes transparent gradients'''
        selected_pts=torch.where(torch.norm(grads,dim=1)>=self.densify_grad_threshold,True,False)
        self.model.split_gaussians(selected_pts)
        if iteration%3000==0:
            self.model.prune_low_opacity(self.opacity_threshold)
    