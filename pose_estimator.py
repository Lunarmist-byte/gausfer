import os
import subprocess
class PoseEstimator:
    """For FR2, Camera Pose estimation using COLMAP, for intrinsic and extrinsic parameter detection"""
    def __init__(self,image_path,output_path):
        self.image_path=image_path
        self.output_path=output_path
    def run_colmap(self):
        '''Uses SfM to get Camera poses'''
        print("Starting Feature Extraction")
        subprocess.run(['colmap','feature_extractor',"--database_path",os.path.join(self.output_path,"database.db"),"--image_path",self.image_path])
        print("Starting Exhaustive Matching")
        subprocess.run(['colmap',"exhaustive_matcher","--database_path",os.path.join(self.output_path,"database.db","--image_path",self.image_path)])
        print("Starting Mapper")
        #for 3D points
        os.makedirs(os.path.join(self.output_path,"sparse"),exist_ok=True)
        subprocess.run(["colmap","mapper","--database_path",os.path.join(self.output_path,"database.db"),"--image_path",self.image_path,"--export_path",os.path.join(self.output_path,"sparse")])
        return "Camera and intrinsic parameters estimated"
