class NeRFTrainer:
    def __init__(self,model,optimizer,train_loader):
        self.model=model.cuda()#for gpu
        self.optimizer=optimizer
        self.train_loader=train_loader
    def train_step(self):
        '''single iteration of training to learn geometry'''
        self.model.train()
        for batch in self.train_loader:
            rays_o,rays_d,target_rgb=batch
            #ray traced volumetric rendering, to ensure high fiedlity
            pass
        