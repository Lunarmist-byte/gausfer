import os
import torch
import numpy as np
from PIL import Image
def qvec2rotmat(qvec):
    '''
    Converts COLMAP's quaternion rotation into a 3x3 Rotation Matrix
    '''
    return np.array([
        [1-2*qvec[2]**2-2*qvec[3]**2,
         2*qvec[1]*qvec[2]-2*qvec[0]*qvec[3],
         2*qvec[3]*qvec[1]+2*qvec[0]*qvec[2]],
         [2*qvec[1]*qvec[2]+2*qvec[0]*qvec[3],
          1-2*qvec[1]**2-2*qvec[3]**2,
          2*qvec[2]*qvec[3]-2*qvec[0]*qvec[1]],
          [2*qvec[3]*qvec[1]-2*qvec[0]*qvec[2],
           2*qvec[2]*qvec[3]+2*qvec[0]*qvec[1],
           1-2*qvec[1]**2-2*qvec[2]**2]
        ])
class RoomDatasetLoader:
    '''
    Parses COLMAP outputs and loads images into memory for NeRF training.
    Fulfils FR1:Upload Multi-Views Images
    '''
    def __init__(self,colmap_dir,images_dir):
        self.colmap_dir=self.colmap_dir
        self.image_dir=images_dir
        self.cameras=self._read_cameras_text(os.path.join(colmap_dir,"cameras.txt"))
        self.images=self._read_images_text(os.path.join(colmap_dir,"images.txt"))
        print(f"Dataset Loaded:{len(self.images)}rooms image ready for training")
    def get_training_batch(self,index):
        '''
        Retrieves a single image and its corresponding pose matrices for the NeRF trainer.
        '''
        img_data=self.images[index]
        cam_data=self.cameras[img_data['camera_id']]
        #load and normalize
        img_path=os.path.join(self.image_dir,img_data['name'])
        image_tensor=torch.tensor(np.array(Image.open(img_path))/255.0,dtype=torch.float32).cuda()
        #Build intrensic matrix
        H,W=cam_data['height'],cam_data['width']
        fx,fy,cx,cy=cam_data['params']
        K=torch.tensor([
            [fx,0,cx],
            [0,fy,cy],
            [0,0,1]
        ],dtype=torch.float32).cuda()
        R=qvec2rotmat(img_data['qvec'])
        T=np.array(img_data['tvec'])
        w2c=np.eye(4)
        w2c[:3,:3]=R
        w2c[:3,3]=T
        c2w=torch.tensor(np.linalg.inv(w2c),dtype=torch.float32).cuda()
        return image_tensor,(H,W,K,c2w)
    def _read_cameras_text(self,path):
        cameras={}
        with open(path,"r") as fid:
            for line in fid:
                line=line.strip()
                if line and line[0]!='#':
                    elems=line.split()
                    cameras[int(elems[0])]={
                        'model':elems[1],'width':int(elems[2]),'height':int(elems[3]),'params':np.array(tuple(map(float,elems[4])))
                    }
        return cameras
    def _read_images_text(self,path):
        images=[]
        with open(path,'r') as fid:
            while True:
                line=fid.readline()
                if not line: break
                if line and line[0]!='#':
                    elems=line.split()
                    images.append({
                        'qvec':np.array(tuple(map(float,elems[1:5]))),
                        'tvec':np.array(tuple(map(float,elems[5:8]))),
                        'camera_id':int(elems[8]),'name':elems[9]
                    })
                    fid.realine()#skips points2d sparse
        return images
    