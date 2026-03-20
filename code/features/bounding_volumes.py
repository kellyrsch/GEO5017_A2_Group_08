def get_axis_aligned_bbox(points):
    aabb = points.get_axis_aligned_bounding_box()
    aabb_volume = aabb.volume()
    return aabb_volume

def get_oriented_bbox(points):
    obb = points.get_oriented_bounding_box()
    obb_volume = obb.volume()
    return obb_volume