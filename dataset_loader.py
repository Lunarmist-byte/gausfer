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
        self.colmap_dir=colmap_dir
        self.image_dir=images_dir
        self.cameras=self._read_cameras_text(os.path.join(colmap_dir,"cameras.txt"))
        self.images=self._read_images_text(os.path.join(colmap_dir,"images.txt"))
        print(f"Dataset Loaded:{len(self.images)}rooms image ready for training")
    def get_training_batch(self,index):
        '''
        Retrieves a single image and its corresponding pose matrices for the NeRF trainer.
        '''
        if not hasattr(self, '_cache'):
            self._cache = {}
        if index in self._cache:
            return self._cache[index]

        img_data=self.images[index]
        cam_data=self.cameras[img_data['camera_id']]
        
        # Load and normalize with robust path recovery
        raw_name = img_data['name']
        # Try primary path
        img_path = os.path.join(self.image_dir, raw_name)
        
        # Robust Recovery: If primary fails, check likely fallback locations
        if not os.path.exists(img_path):
            # Fallback 1: Check in ./images/ folder relative to project root
            fallback_images = os.path.join(os.getcwd(), 'images', os.path.basename(raw_name))
            # Fallback 2: Check in current directory root
            fallback_root = os.path.join(os.getcwd(), os.path.basename(raw_name))
            
            if os.path.exists(fallback_images):
                img_path = fallback_images
            elif os.path.exists(fallback_root):
                img_path = fallback_root
            else:
                # If everything fails, raise a clear error with suggestions
                raise FileNotFoundError(f"Could not find image '{raw_name}' in '{self.image_dir}' or fallbacks. "
                                        f"Please ensure your images are in the './images' folder.")

        img_pil = Image.open(img_path).convert('RGB')
        
        # Initial intrinsic matrix parameters
        H,W=cam_data['height'],cam_data['width']
        fx,fy,cx,cy=cam_data['params']
        
        # Performance Hint: Downsample if image is too large (e.g., 4K)
        max_dim = 1200
        if max(H, W) > max_dim:
            scale = max_dim / max(H, W)
            new_W, new_H = int(W * scale), int(H * scale)
            img_pil = img_pil.resize((new_W, new_H), Image.Resampling.LANCZOS)
            fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale
            H, W = new_H, new_W

        image_tensor=torch.tensor(np.array(img_pil)/255.0,dtype=torch.float32).cuda()
        
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
        
        self._cache[index] = (image_tensor, (H, W, K, c2w))
        return self._cache[index]
    def _read_cameras_text(self,path):
        cameras={}
        with open(path,"r") as fid:
            for line in fid:
                line=line.strip()
                if line and line[0]!='#':
                    elems=line.split()
                    cameras[int(elems[0])]={
                        'model':elems[1],'width':int(elems[2]),'height':int(elems[3]),'params':np.array(tuple(map(float,elems[4:])))
                    }
        return cameras
    def _read_images_text(self,path):
        images=[]
        with open(path,'r') as fid:
            while True:
                line=fid.readline()
                if not line: break
                line = line.strip()
                if line and line[0]!='#':
                    elems=line.split()
                    images.append({
                        'qvec':np.array(tuple(map(float,elems[1:5]))),
                        'tvec':np.array(tuple(map(float,elems[5:8]))),
                        'camera_id':int(elems[8]),'name':elems[9]
                    })
                    fid.readline()#skips points2d sparse
        return images
    