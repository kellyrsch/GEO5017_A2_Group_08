from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import os

from feature_selection import plot_feature_distribution, select_features_based_on_J_score
from features.number_of_points import calculate_number_of_points
from features.convex_hull import compute_convex_hull_volume
from features.convex_hull import point_density_in_convex_hull
from features.bounding_volumes import get_axis_aligned_bbox
from features.bounding_volumes import get_oriented_bbox
from features.bounding_volumes import get_height_of_aa_bbox
from features.height_width_ratio import height_width_ratio
from features.footprint_area import footprint_area
from features.lps_features import lps_features

def get_linearity(points):
    return lps_features(points)[0]

def get_planarity(points):
    return lps_features(points)[1]

def get_scattering(points):
    return lps_features(points)[2]

FOLDERNAME = os.path.join(Path(os.path.dirname(__file__)).parent.absolute(), "pointclouds-500")

LABELS = {
    100: "building",
    200: "car",
    300: "fence",
    400: "pole",
    500: "tree"
}

def load_pts_with_labels(index_start: int, index_end: int):
    pointcloud = []
    i = index_start
    while i <= index_end:
        filename = f"{i:03d}.xyz"
        filepath = os.path.join(FOLDERNAME, filename)
        pcd = o3d.io.read_point_cloud(filepath)
        pointcloud.append((pcd, LABELS.get(100 + i // 100 * 100, "unknown")))
        i += 1
    return pointcloud

def apply_train_test_split(samples_with_labels: list[tuple[object, str]], test_size: float = 0.2, random_seed: int = 42) -> tuple[list[tuple[object, str]], list[tuple[object, str]]]:
    np.random.seed(random_seed)
    samples_per_label = defaultdict(list)
    for sample, label in samples_with_labels:
        samples_per_label[label].append(sample)
    train_samples = []
    test_samples = []
    for label, samples in samples_per_label.items():
        np.random.shuffle(samples)
        split_index = int(len(samples) * (1 - test_size))
        train_samples.extend([(s, label) for s in samples[:split_index]])
        test_samples.extend([(s, label) for s in samples[split_index:]])
    return train_samples, test_samples

point_clouds_with_labels = load_pts_with_labels(0, 499)

train_samples, test_samples = apply_train_test_split(point_clouds_with_labels, test_size=0.33)

features = [
    (calculate_number_of_points, "Number of Points"),
    (compute_convex_hull_volume, "Volume of Convex Hull"),
    (point_density_in_convex_hull, "Point density in convex hull"),
    (get_axis_aligned_bbox, "Volume of axis aligned bbox"),
    (get_oriented_bbox, "Volume of oriented bbox"),
    (get_height_of_aa_bbox, "Extent in y-dimension (height) of axis aligned bbox"),
    (height_width_ratio, "Height-to-width ratio"),
    (footprint_area, "Top-down footprint area"),
    (get_linearity, "Linearity"),
    (get_planarity, "Planarity"),
    (get_scattering, "Scattering")
]

#plot_feature_distribution(point_clouds_with_labels, features)
features_to_use = select_features_based_on_J_score(train_samples, features, desired_feature_count=4)