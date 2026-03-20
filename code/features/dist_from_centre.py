import numpy as np

def calculate_distance_from_centre(pcd):
    return np.mean(np.linalg.norm(np.array(pcd.points) - np.mean(np.array(pcd.points), axis=0), axis=1))