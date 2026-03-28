import numpy as np

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