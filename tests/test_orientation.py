import math

import numpy as np

from reconstruction.cab_model import box_corners
from reconstruction.camera import Camera
from reconstruction.orientation import (
    OrientationSettings,
    SilhouetteMatcher,
    estimate_heading,
    mask_axis,
    polygon_moments,
)
from scipy.spatial import ConvexHull

DIMS = np.array([1650.0, 255.0, 400.0])
CENTER = np.array([0.0, 0.0, 200.0])


def look_at(position, target, f=1600.0):
    d = np.asarray(target, float) - np.asarray(position, float)
    return Camera(position, 0.0, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]),
                  f, f, 960, 540, 1920, 1080)


CAMS = [look_at((-5000, -4000, 1500), CENTER), look_at((5000, -3500, 1800), CENTER),
        look_at((3000, 5000, 1200), CENTER)]


def silhouette(cam, heading, dims=DIMS, center=CENTER):
    d = np.array([math.cos(heading), math.sin(heading), 0.0])
    ref = center + dims[0] / 2 * d - np.array([0, 0, dims[2] / 2])
    uv = cam.project(box_corners(ref[None], d[None], dims)[0])
    return uv[ConvexHull(uv).vertices]


def wrap_deg(a):
    return (a + 180) % 360 - 180


def test_polygon_moments_and_mask_axis_of_rotated_rectangle():
    a = math.radians(30)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    rect = np.array([[-100, -20], [100, -20], [100, 20], [-100, 20]], float) @ R.T + [500, 300]
    area, c, _ = polygon_moments(rect)
    assert area == np.float64(200 * 40) or abs(area - 8000) < 1e-6
    assert np.allclose(c, [500, 300])
    angle, elong = mask_axis(rect)
    assert abs(wrap_deg(math.degrees(angle) - 30) % 180) < 1e-6 or abs(abs(wrap_deg(math.degrees(angle) - 30)) - 180) < 1e-6
    assert elong == np.float64(5.0) or abs(elong - 5.0) < 1e-6


def test_axis_recovered_and_front_back_kept_ambiguous_for_symmetric_box():
    matcher = SilhouetteMatcher(OrientationSettings(), None, DIMS)
    truth = math.radians(35)
    views = [(i, c, silhouette(c, truth), 1.0) for i, c in enumerate(CAMS)]
    m = estimate_heading(matcher, views, CENTER)
    assert m.ok
    err = wrap_deg(math.degrees(m.heading - truth))
    # коробка симметрична: направление не определено (axis_only), но ось точна (с точностью до 180°)
    assert m.axis_only
    assert min(abs(err), abs(abs(err) - 180)) < 4.0
    assert abs(m.front_margin) < OrientationSettings().min_front_margin


def test_heading_independent_of_center_error():
    """Совмещение по центроиду: ошибка центра на 2 м почти не сдвигает курс."""
    matcher = SilhouetteMatcher(OrientationSettings(), None, DIMS)
    truth = math.radians(-60)
    views = [(i, c, silhouette(c, truth), 1.0) for i, c in enumerate(CAMS)]
    m = estimate_heading(matcher, views, CENTER + np.array([200.0, -150.0, 0.0]))
    err = wrap_deg(math.degrees(m.heading - truth))
    assert m.ok and min(abs(err), abs(abs(err) - 180)) < 6.0


def test_garbage_or_tiny_mask_is_not_used():
    """Маска, не похожая на модель ни при каком курсе (IoU < min_iou), и слишком маленькая — не используются.
    Частично закрытую маску сопоставитель отличить не может (она похожа на вид спереди): такие камеры
    исключает трекер по закрытым рёбрам рамки (см. test_upgrade_tracker.py)."""
    matcher = SilhouetteMatcher(OrientationSettings(), None, DIMS)
    c = silhouette(CAMS[0], 0.3).mean(axis=0)
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    star = c + np.c_[np.cos(t), np.sin(t)] * np.where(np.arange(40) % 2, 150.0, 20.0)[:, None]
    assert not estimate_heading(matcher, [(0, CAMS[0], star, 1.0)], CENTER).ok
    tiny = c + np.array([[0, 0], [10, 0], [10, 8], [0, 8]], float)
    assert not estimate_heading(matcher, [(0, CAMS[0], tiny, 1.0)], CENTER).ok


def test_conflicting_camera_does_not_dominate():
    matcher = SilhouetteMatcher(OrientationSettings(), None, DIMS)
    truth = math.radians(20)
    views = [(0, CAMS[0], silhouette(CAMS[0], truth), 1.0), (1, CAMS[1], silhouette(CAMS[1], truth), 1.0),
             (2, CAMS[2], silhouette(CAMS[2], truth + math.radians(40)), 1.0)]
    m = estimate_heading(matcher, views, CENTER)
    err = wrap_deg(math.degrees(m.heading - truth))
    assert m.ok and min(abs(err), abs(abs(err) - 180)) < 10.0


def test_no_masks_gives_no_measurement():
    matcher = SilhouetteMatcher(OrientationSettings(), None, DIMS)
    m = estimate_heading(matcher, [], CENTER)
    assert not m.ok and m.reason
