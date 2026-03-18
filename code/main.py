import numpy as np
import open3d as o3d
import os

foldername = "pointclouds-500"

# read the pointclouds from specified indices and put them in a list
def load_pts(index_start, index_end, folder):
    pointcloud = []
    i = index_start
    while i <= index_end:
        filename = f"{i:03d}.xyz"
        print(filename)
        filepath = os.path.join(folder, filename)
        print(filepath)
        pcd = o3d.io.read_point_cloud(filepath)
        print(pcd)
        pointcloud.append(pcd)
        i += 1
    return pointcloud

pointcloud = load_pts(0, 5, foldername)
print(pointcloud)

