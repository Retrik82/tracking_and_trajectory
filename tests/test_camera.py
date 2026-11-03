import math

import numpy as np
import pytest

from reconstruction.camera import Camera, camera_axes, load_camera_poses
from tests.conftest import camera_csv, load_target_positions, make_camera, requires_data


def spec_projection(P, C, r, p, y, fx, fy, cx=960.0, cy=540.0):
    """Проекция, записанная буквально по формулам модели камеры, без матриц."""
    forward = [math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)]
    right = [math.sin(r) * math.sin(p) * math.cos(y) - math.cos(r) * math.sin(y),
             math.sin(r) * math.sin(p) * math.sin(y) + math.cos(r) * math.cos(y),
             -math.sin(r) * math.cos(p)]
    up = [-(math.cos(r) * math.sin(p) * math.cos(y) + math.sin(r) * math.sin(y)),
          math.cos(y) * math.sin(r) - math.cos(r) * math.sin(p) * math.sin(y),
          math.cos(r) * math.cos(p)]
    d = [P[i] - C[i] for i in range(3)]
    dot = lambda a, b: sum(a[i] * b[i] for i in range(3))
    z = dot(d, forward)
    return cx + fx * dot(d, right) / z, cy - fy * dot(d, up) / z


@pytest.mark.parametrize("angles_deg", [(0, 0, 0), (-4.4117, -6.9741, -164.0), (-0.6776, -3.114, 115.1096), (10, 20, 30)])
def test_axes_orthonormal(angles_deg):
    forward, right, up = camera_axes(*np.radians(angles_deg))
    M = np.vstack([forward, right, up])
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)


def test_zero_angles_axes_match_unreal():
    # В UE при нулевых углах камера смотрит вдоль +X, вправо — +Y, вверх — +Z
    forward, right, up = camera_axes(0, 0, 0)
    np.testing.assert_allclose(forward, [1, 0, 0])
    np.testing.assert_allclose(right, [0, 1, 0])
    np.testing.assert_allclose(up, [0, 0, 1])


def test_project_matches_spec_formulas():
    rng = np.random.default_rng(0)
    C = np.array([100.0, -200.0, 1500.0])
    r, p, y = np.radians([-4.4117, -6.9741, -164.0])
    cam = Camera(C, r, p, y, f_x=1700.0, f_y=1600.0, c_x=960, c_y=540, width=1920, height=1080)
    forward = cam.forward
    for _ in range(50):
        P = C + forward * rng.uniform(1000, 30000) + rng.normal(0, 3000, 3)
        if np.dot(P - C, forward) <= 0:
            continue
        expected = spec_projection(P, C, r, p, y, 1700.0, 1600.0)
        np.testing.assert_allclose(cam.project(P), expected, atol=1e-6)


def test_point_behind_camera_is_nan():
    cam = Camera((0, 0, 0), 0, 0, 0, 1000, 1000, 960, 540, 1920, 1080)
    assert np.all(np.isnan(cam.project([-100.0, 0, 0])))


def test_yaw_wraparound_is_continuous():
    # yaw = +179.9° и -179.9° — соседние направления; проекция должна меняться плавно
    P = np.array([-10000.0, 50.0, 0.0])
    a = Camera((0, 0, 0), 0, 0, math.radians(179.9), 1000, 1000, 960, 540, 1920, 1080).project(P)
    b = Camera((0, 0, 0), 0, 0, math.radians(-179.9), 1000, 1000, 960, 540, 1920, 1080).project(P)
    assert np.hypot(*(a - b)) < 5.0


@requires_data
def test_csv_reads_cp1251_and_all_frames():
    poses = load_camera_poses(str(camera_csv(5)))
    assert sorted(poses) == list(range(1002))
    _, _, _, yaw = poses[0]
    assert math.degrees(yaw) == pytest.approx(-164.0)


@requires_data
def test_camera5_projection_through_yaw_wrap():
    # Между кадрами 623 и 624 yaw Camera5 переходит с -179.8° на +179.7°.
    # Проекция цели должна сдвигаться на величину порядка межкадрового движения, а не скачком.
    target = load_target_positions()
    prev = None
    for frame in range(615, 635):
        uv = make_camera(5, frame).project(np.array(target[frame]))
        assert np.all(np.isfinite(uv))
        if prev is not None:
            assert np.hypot(*(uv - prev)) < 40.0, f"скачок проекции на кадре {frame}"
        prev = uv
