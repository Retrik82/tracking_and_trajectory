import math

import numpy as np

from reconstruction.cab_model import box_corners
from reconstruction.camera import Camera
from reconstruction.silhouette_fit import SilhouetteFitSettings, fit_pose, select_views

DIMS = np.array([1650.0, 255.0, 400.0])


def look_at(position, target, f=1600.0):
    d = np.asarray(target, float) - np.asarray(position, float)
    return Camera(position, 0.0, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]), f, f, 960, 540,
                  1920, 1080)


def mask(cam, center, heading):
    d = np.array([math.cos(heading), math.sin(heading), 0.0])
    ref = center + DIMS[0] / 2 * d - np.array([0, 0, DIMS[2] / 2])
    return cam.project(box_corners(ref[None], d[None], DIMS)[0])


def test_fit_recovers_pose_from_two_cameras_and_skips_single_or_occluded():
    center, heading = np.array([300.0, -200.0, 200.0]), math.radians(25)
    cams = {0: look_at((-5000, -4000, 1500), center), 1: look_at((5000, -3500, 1800), center)}
    masks = {c: mask(cam, center, heading) for c, cam in cams.items()}
    s = SilhouetteFitSettings(prior_sigma=500.0, prior_sigma_heading_deg=30.0, area_heading_tol_deg=15.0)
    start, h0 = center + np.array([120.0, -90.0, 0.0]), heading + math.radians(8)
    views = select_views(cams, masks, start, h0, 0.0, None, DIMS, s)
    fit = fit_pose(views, start, h0, 0.0, None, DIMS, s)
    assert fit.ok
    assert math.hypot(fit.x - center[0], fit.y - center[1]) < 25.0                  # < 25 см
    assert abs(math.degrees(fit.heading - heading)) < 1.5
    # одна камера — не используется (глубина не определена)
    assert not fit_pose(views[:1], start, h0, 0.0, None, DIMS, s).ok
    # маска, закрытая наполовину, не проходит проверку площади
    c = masks[1].mean(axis=0)
    half = {0: masks[0], 1: c + (masks[1] - c) * np.array([0.4, 1.0])}
    assert len(select_views(cams, half, start, h0, 0.0, None, DIMS, s)) == 1
