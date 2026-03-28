import numpy as np
from scipy.spatial import ConvexHull

def footprint_area(points):
    pts = np.asarray(points.points)
    xy = pts[:, :2]

    if len(xy) < 3:
        return 0.0

    hull = ConvexHull(xy)
    return hull.volume

def get_axis_aligned_bbox(points):
    aabb = points.get_axis_aligned_bounding_box()
    aabb_volume = aabb.volume()
    return aabb_volume

def get_oriented_bbox(points):
    obb = points.get_oriented_bounding_box()
    obb_volume = obb.volume()
    return obb_volume

def get_height_of_aa_bbox(points):
    aabb = points.get_axis_aligned_bounding_box()
    aabb_extent = aabb.get_extent()
    aabb_height = aabb_extent[1]
    return aabb_height

def compute_convex_hull_volume(points):
    hull = points.compute_convex_hull()
    hull = hull[0]
    hull.orient_triangles()
    hull_volume = hull.get_volume()
    return hull_volume

def point_density_in_convex_hull(points):
    hull_volume = compute_convex_hull_volume(points)
    nr_of_pts = len(points.points)
    point_density = nr_of_pts / hull_volume
    return point_density

def height_width_ratio(points):
    pts = np.asarray(points.points)

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    extents = maxs - mins

    width = extents[0]
    height = extents[2]

    return height / (width + 1e-8)

def lps_features(points):
    pts = np.asarray(points.points)

    centered = pts - np.mean(pts, axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvals(cov))[::-1]

    l1, l2, l3 = eigvals

    linearity = (l1 - l2) / (l1 + 1e-8)
    planarity = (l2 - l3) / (l1 + 1e-8)
    scattering = l3 / (l1 + 1e-8)

    return linearity, planarity, scattering

def get_linearity(points):
    return lps_features(points)[0]

def get_planarity(points):
    return lps_features(points)[1]

def get_scattering(points):
    return lps_features(points)[2]