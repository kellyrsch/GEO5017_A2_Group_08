def compute_convex_hull_volume(points):
    hull = points.compute_convex_hull()
    hull = hull[0]
    hull.orient_triangles()
    hull_volume = hull.get_volume()
    return hull_volume
