import numpy as np
from scipy.spatial import ConvexHull

def footprint_area(points):
    pts = np.asarray(points.points)
    xy = pts[:, :2]

    if len(xy) < 3:
        return 0.0

    hull = ConvexHull(xy)
    return hull.volume