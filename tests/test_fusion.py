import math

import numpy as np

from reconstruction.cab_model import TruckDimensions
from reconstruction.camera import Camera
from reconstruction.fusion import (
    FusionSettings, build_point_measurement, centroid_offset, point_covariance, projection_jacobian,
    ray_angles_deg, silhouette_points, triangulate_realtime,
)

DIMS = TruckDimensions(1650.0, 255.0, 400.0).as_array()


def look_at(position, target, f=1500.0):
    d = np.asarray(target, float) - np.asarray(position, float)
    return Camera(position, 0.0, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]),
                  f, f, 960, 540, 1920, 1080)


TARGET = np.array([2000.0, 500.0, 300.0])
CAMS = {1: look_at((-3000, -7000, 1500), TARGET), 2: look_at((9000, -6000, 1800), TARGET),
        3: look_at((5000, 8000, 1200), TARGET)}


def test_projection_jacobian_matches_finite_differences():
    cam = CAMS[1]
    J = projection_jacobian(cam, TARGET)
    for k in range(3):
        d = np.zeros(3)
        d[k] = 1.0
        num = (cam.project(TARGET + d) - cam.project(TARGET - d)) / 2
        np.testing.assert_allclose(J[:, k], num, rtol=1e-4, atol=1e-6)


def test_covariance_shrinks_with_more_cameras_and_grows_with_noise():
    c2 = point_covariance([CAMS[1], CAMS[2]], TARGET, 5.0)
    c3 = point_covariance([CAMS[1], CAMS[2], CAMS[3]], TARGET, 5.0)
    assert np.trace(c3) < np.trace(c2)
    assert np.trace(point_covariance([CAMS[1], CAMS[2]], TARGET, 10.0)) > np.trace(c2) * 3


def test_triangulate_realtime_ok_and_rejects_bad_geometry():
    ids = [1, 2, 3]
    px = [tuple(CAMS[i].project(TARGET)) for i in ids]
    m = triangulate_realtime([CAMS[i] for i in ids], ids, px, FusionSettings())
    assert m.ok and np.linalg.norm(m.point - TARGET) < 1.0 and m.cameras == ids and m.std_m < 0.5
    # почти параллельные лучи: две камеры в одном месте
    near = look_at((-3000, -7000, 1500), TARGET)
    near2 = look_at((-3001, -7000, 1500), TARGET)
    m2 = triangulate_realtime([near, near2], [1, 2], [tuple(near.project(TARGET)), tuple(near2.project(TARGET))],
                              FusionSettings())
    assert not m2.ok and "параллельны" in m2.reason
    # выброс на одной камере отбрасывается перебором
    px_bad = list(px)
    px_bad[2] = (px[2][0] + 300, px[2][1])
    m3 = triangulate_realtime([CAMS[i] for i in ids], ids, px_bad, FusionSettings())
    assert m3.ok and 3 not in m3.cameras and np.linalg.norm(m3.point - TARGET) < 1.0
    assert ray_angles_deg([CAMS[1], CAMS[2]], TARGET).min() > 10


def test_model_corrected_centroids_remove_view_bias():
    # «центроид маски» = центроид силуэта коробки на каждом ракурсе — разные точки кузова
    direction = np.array([1.0, 0.3, 0.0])
    centroids = {}
    for cid, cam in CAMS.items():
        uv = cam.project(silhouette_points(TARGET, direction, None, DIMS))
        centroids[cid] = tuple(uv.mean(axis=0))      # грубый центроид (среднее углов)
    raw = build_point_measurement(CAMS, centroids, FusionSettings(mode="centroid"))
    corrected = build_point_measurement(CAMS, centroids, FusionSettings(mode="model_corrected"),
                                        predicted_center=TARGET + [150, -80, 20], direction=direction, dims=DIMS)
    assert raw.ok and corrected.ok
    err_raw = np.linalg.norm(raw.point - TARGET)
    err_corr = np.linalg.norm(corrected.point - TARGET)
    assert err_corr < err_raw and err_corr < 150            # см; прогноз смещён на 1.7 м, поправка всё равно помогает
    assert centroid_offset(CAMS[1], TARGET, direction, None, DIMS) is not None
