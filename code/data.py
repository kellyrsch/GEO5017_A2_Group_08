from collections import defaultdict
import os
from pathlib import Path
import numpy as np
import open3d as o3d


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

def get_data(test_size: float, seed: int = 42):
    pointclouds_with_labels = load_pts_with_labels(0, 499)
    return apply_train_test_split(pointclouds_with_labels, test_size=test_size, random_seed=seed)

def get_data_for_sklearn(training_samples: list[tuple[object, str]], testing_samples: list[tuple[object, str]], feature_classifiers: list[tuple[callable, str]]):
    x_train = []
    y_train = []
    for sample, label in training_samples:
        features = [classifier(sample) for classifier, _ in feature_classifiers]
        x_train.append(features)
        y_train.append(label)

    x_test = []
    y_test = []
    for sample, label in testing_samples:
        features = [classifier(sample) for classifier, _ in feature_classifiers]
        x_test.append(features)
        y_test.append(label)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)