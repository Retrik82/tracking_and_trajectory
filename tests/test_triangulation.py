import math

import numpy as np
import pytest

from reconstruction.camera import Camera
from reconstruction.triangulation import (
    calculate_ray_error,
    pixel_to_ray,
    reprojection_errors,
    triangulate,
    triangulate_robust,
)
from tests.conftest import load_target_positions, make_camera, requires_data

TRUE_POINT = np.array([500.0, 300.0, 200.0])


def look_at_camera(position, target, f=1500.0):
    """Камера в position, направленная на target (roll = 0)."""
    d = np.asarray(target, float) - np.asarray(position, float)
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
    return Camera(position, 0.0, pitch, yaw, f, f, 960, 540, 1920, 1080)


@pytest.fixture
def cameras():
    positions = [(-5000, 0, 1000), (5000, 1000, 1500), (0, -6000, 800), (1000, 7000, 2000)]
    return [look_at_camera(p, TRUE_POINT + [200, -100, 50]) for p in positions]


def test_pixel_to_ray_inverts_projection(cameras):
    for cam in cameras:
        u, v = cam.project(TRUE_POINT)
        ray = pixel_to_ray(cam, u, v)
        expected = (TRUE_POINT - cam.position) / np.linalg.norm(TRUE_POINT - cam.position)
        np.testing.assert_allclose(ray, expected, atol=1e-12)


def test_triangulate_noise_free(cameras):
    detections = [tuple(c.project(TRUE_POINT)) for c in cameras]
    point = triangulate(cameras, detections)
    np.testing.assert_allclose(point, TRUE_POINT, atol=1e-6)
    assert calculate_ray_error(point, cameras, detections) < 1e-6
    assert reprojection_errors(point, cameras, detections).max() < 1e-6


def test_triangulate_uses_all_cameras(cameras):
    # Раньше return стоял внутри цикла и решалась система только по первой камере
    detections = [tuple(c.project(TRUE_POINT)) for c in cameras]
    detections[0] = (detections[0][0] + 30, detections[0][1])
    point_all = triangulate(cameras, detections)
    point_tail = triangulate(cameras[1:], detections[1:])
    assert np.linalg.norm(point_all - point_tail) > 1e-3


def test_triangulate_with_noise(cameras):
    rng = np.random.default_rng(42)
    errors_3d, errors_px = [], []
    for _ in range(200):
        detections = [tuple(c.project(TRUE_POINT) + rng.normal(0, 1.0, 2)) for c in cameras]
        result = triangulate_robust(cameras, detections, max_reprojection_error_px=10)
        assert result.ok and result.num_inliers == 4
        errors_3d.append(np.linalg.norm(result.point - TRUE_POINT))
        errors_px.append(result.reprojection_errors.max())
    # Шум 1 px на дистанции ~5 км при f = 1500 px: ~3 см на камеру, 4 камеры дают меньше
    assert np.median(errors_3d) < 5.0
    assert np.max(errors_px) < 5.0


def test_robust_drops_outlier_camera(cameras):
    detections = [tuple(c.project(TRUE_POINT)) for c in cameras]
    detections[2] = (detections[2][0] + 150, detections[2][1] - 80)
    result = triangulate_robust(cameras, detections, max_reprojection_error_px=10)
    assert result.ok
    assert result.inliers.tolist() == [True, True, False, True]
    np.testing.assert_allclose(result.point, TRUE_POINT, atol=1e-6)
    assert result.reprojection_errors[2] > 100  # ошибка выброса сохраняется для отчёта


def test_robust_two_cameras_cannot_resolve_outlier(cameras):
    detections = [tuple(c.project(TRUE_POINT)) for c in cameras[:2]]
    detections[1] = (detections[1][0] + 150, detections[1][1])
    result = triangulate_robust(cameras[:2], detections, max_reprojection_error_px=10)
    assert not result.ok
    assert result.inliers.all()


def test_parallel_rays_rejected():
    a = Camera((0, 0, 0), 0, 0, 0, 1000, 1000, 960, 540, 1920, 1080)
    b = Camera((0, 1e-9, 0), 0, 0, 0, 1000, 1000, 960, 540, 1920, 1080)
    with pytest.raises(np.linalg.LinAlgError):
        triangulate([a, b], [(960, 540), (960, 540)])
    assert not triangulate_robust([a, b], [(960, 540), (960, 540)]).ok


def test_single_camera():
    a = Camera((0, 0, 0), 0, 0, 0, 1000, 1000, 960, 540, 1920, 1080)
    with pytest.raises(ValueError):
        triangulate([a], [(960, 540)])
    assert not triangulate_robust([a], [(960, 540)]).ok


@requires_data
def test_real_cameras_recover_target_from_its_projections():
    # Проецируем цель из CSV во все камеры, где она попадает в кадр, и триангулируем обратно.
    # Проверяет согласованность геометрии на настоящих позах, включая поворот Camera5 через ±180°.
    target = load_target_positions()
    checked = 0
    for frame in list(range(0, 1001, 25)) + list(range(615, 635)):
        if frame not in target:
            continue
        P = np.array(target[frame])
        cams, dets = [], []
        for cam_id in range(1, 6):
            cam = make_camera(cam_id, frame)
            uv = cam.project(P)
            if np.all(np.isfinite(uv)) and 0 <= uv[0] < 1920 and 0 <= uv[1] < 1080:
                cams.append(cam)
                dets.append(tuple(uv))
        if len(cams) < 2:
            continue
        result = triangulate_robust(cams, dets)
        assert result.ok, frame
        assert np.linalg.norm(result.point - P) < 1e-3, frame  # см
        checked += 1
    assert checked > 40
