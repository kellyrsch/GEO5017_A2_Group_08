import numpy as np

def height_width_ratio(points):
    pts = np.asarray(points.points)

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    extents = maxs - mins

    width = extents[0]
    height = extents[2]

    return height / (width + 1e-8)