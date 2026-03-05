import os
import subprocess
class PoseEstimator:
    """For FR2,Use CUDA Camera Pose estimation using COLMAP, for intrinsic and extrinsic parameter detection"""
    def __init__(self,image_path,output_path):
        self.image_path=image_path
        self.output_path=output_path
    def run_colmap(self):
        '''Uses SfM to get Camera poses'''
        db_path=os.path.join(self.output_path,"database.db")
        sparse_path=os.path.join(self.output_path,"sparse")
        print("Starting Feature Extraction with CUDA")
        subprocess.run([
            'colmap','feature_extractor',
            '--database_path',db_path,
            '--image_path',self.image_path
        ],check=True)
        print("Starting Sequential Matching with Loop Detection with CUDA")
        subprocess.run([
            'colmap','sequential_matcher',
            '--database_path',db_path,
            '--SequentialMatching.loop_detection','1'#close within parameters
        ],check=True)
        print("Starting Mapper(SfM)..")
        os.makedirs(sparse_path,exist_ok=True)
        subprocess.run([
            'colmap','mapper',
            '--database_path',db_path,
            '--image_path',self.image_path,
            '--export_path',sparse_path
        ],check=True)
        return "Room camera poses and intrinsic parameters estimated successfully"

