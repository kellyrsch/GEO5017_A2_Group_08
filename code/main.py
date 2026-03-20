import numpy as np
import open3d as o3d
import os

from feature_distribution import get_feature_stats_over_labels
from features.number_of_points import calculate_number_of_points
from features.convex_hull import compute_convex_hull_volume
from features.bounding_volumes import get_axis_aligned_bbox
from features.bounding_volumes import get_oriented_bbox

FOLDERNAME = os.path.join(os.path.dirname(__file__), "../pointclouds-500")

LABELS = {
    100: "building",
    200: "car",
    300: "fence",
    400: "pole",
    500: "tree"
}

def load_pts_with_labels(index_start, index_end):
    pointcloud = []
    i = index_start
    while i <= index_end:
        filename = f"{i:03d}.xyz"
        filepath = os.path.join(FOLDERNAME, filename)
        pcd = o3d.io.read_point_cloud(filepath)
        pointcloud.append((pcd, LABELS.get(100 + i // 100 * 100, "unknown")))
        i += 1
    return pointcloud

point_clouds_with_labels = load_pts_with_labels(0, 499)

#print(point_clouds_with_labels)

get_feature_stats_over_labels(point_clouds_with_labels, [
    (calculate_number_of_points, "Number of Points"),
    (compute_convex_hull_volume, "Volume of Convex Hull"),
    (get_axis_aligned_bbox, "Volume of axis aligned bbox"),
    (get_oriented_bbox, "Volume of oriented bbox")
])