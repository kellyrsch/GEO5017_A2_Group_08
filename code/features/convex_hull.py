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
