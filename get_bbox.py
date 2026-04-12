import numpy as np
import os

def get_colmap_bbox(sparse_path):
    points_path = os.path.join(sparse_path, "points3D.txt")
    if not os.path.exists(points_path):
        return None
        
    xyz = []
    with open(points_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    if not xyz:
        return None
        
    xyz = np.array(xyz)
    p_min = np.min(xyz, axis=0)
    p_max = np.max(xyz, axis=0)
    
    # Add some padding (10%)
    center = (p_min + p_max) / 2
    extent = (p_max - p_min) * 1.1
    p_min = center - extent / 2
    p_max = center + extent / 2
    
    return p_min, p_max

if __name__ == "__main__":
    sparse_dir = r"c:\Users\amals\Documents\Codeworks\gausfer\output\sparse\0"
    bbox = get_colmap_bbox(sparse_dir)
    if bbox:
        p_min, p_max = bbox
        print(f"BBOX_MIN: {p_min.tolist()}")
        print(f"BBOX_MAX: {p_max.tolist()}")
    else:
        print("No points found.")
