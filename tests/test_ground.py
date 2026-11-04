import math

import numpy as np
import pytest

from reconstruction.camera import Camera
from reconstruction.ground import (
    OnlineRoadModel,
    RoadPlane,
    ground_homography,
    ground_jacobian,
    pixel_to_ground,
    pixel_to_plane,
    plane_to_pixel,
)


def look_at(position, target, f=1500.0, roll=0.0):
    d = np.asarray(target, float) - np.asarray(position, float)
    return Camera(position, roll, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]),
                  f, f * 0.98, 960, 540, 1920, 1080)


CAMS = [look_at((-3000, -7000, 1500), (0, 0, 0)), look_at((9000, -6000, 1800), (500, 200, 0)),
        look_at((5000, 8000, 1200), (0, 0, 0), roll=math.radians(7))]


@pytest.mark.parametrize("cam", CAMS)
def test_homography_matches_projection_and_inverts(cam):
    plane = RoadPlane.horizontal(-35.0)
    h = ground_homography(cam, plane)
    assert h.ok
    rng = np.random.default_rng(0)
    world = np.c_[rng.uniform(-2000, 2000, (20, 2)), np.full(20, -35.0)]
    uv_true = cam.project(world)
    uv = plane_to_pixel(h, plane.to_plane(world))
    assert np.allclose(uv, uv_true, atol=1e-6)
    back, ok = pixel_to_ground(h, uv_true)
    assert ok.all()
    assert np.allclose(back, world, atol=1e-4)


def test_tilted_plane_round_trip():
    plane = RoadPlane(np.array([100.0, -50.0, 20.0]), np.array([0.05, -0.08, 1.0]))
    cam = CAMS[0]
    h = ground_homography(cam, plane)
    ab = np.array([[0.0, 0.0], [800.0, -300.0], [-1200.0, 600.0]])
    world = plane.from_plane(ab)
    # точки действительно на плоскости
    assert np.allclose((world - plane.point) @ plane.normal, 0.0, atol=1e-9)
    uv = plane_to_pixel(h, ab)
    assert np.allclose(uv, cam.project(world), atol=1e-6)
    back, ok = pixel_to_plane(h, uv)
    assert ok.all() and np.allclose(back, ab, atol=1e-4)
    assert plane.height_at(world[1, :2]) == pytest.approx(world[1, 2], abs=1e-9)


def test_degenerate_camera_in_plane_is_rejected():
    plane = RoadPlane.horizontal(0.0)
    cam = Camera((0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 1500, 1500, 960, 540, 1920, 1080)   # центр камеры в плоскости
    h = ground_homography(cam, plane)
    assert not h.ok
    _, ok = pixel_to_plane(h, np.array([[960.0, 700.0]]))
    assert not ok.any()


def test_pixel_above_horizon_and_beyond_work_area_are_invalid():
    cam = CAMS[0]
    h = ground_homography(cam, RoadPlane.horizontal(0.0))
    # верх кадра камеры, смотрящей чуть вниз, — небо: луч не пересекает дорогу перед камерой
    _, ok = pixel_to_plane(h, np.array([[960.0, 0.0]]))
    assert not ok[0]
    near = cam.project(np.array([0.0, 0.0, 0.0]))
    _, ok = pixel_to_plane(h, near[None, :], max_range=1000.0)    # до точки ~78 м > 10 м
    assert not ok[0]
    _, ok = pixel_to_plane(h, near[None, :], max_range=20000.0)
    assert ok[0]


def test_ground_jacobian_matches_finite_difference():
    cam = CAMS[1]
    h = ground_homography(cam, RoadPlane.horizontal(0.0))
    uv = cam.project(np.array([300.0, 100.0, 0.0]))
    J = ground_jacobian(h, uv)
    eps = 0.5
    num = np.c_[(pixel_to_ground(h, (uv + [eps, 0])[None])[0][0] - pixel_to_ground(h, (uv - [eps, 0])[None])[0][0]) / (2 * eps),
                (pixel_to_ground(h, (uv + [0, eps])[None])[0][0] - pixel_to_ground(h, (uv - [0, eps])[None])[0][0]) / (2 * eps)]
    assert np.allclose(J, num, rtol=1e-3, atol=1e-6)


def test_homography_consistent_between_cameras():
    """Точка дороги, найденная по пикселю одной камеры, проецируется во вторую туда же, где её видит вторая."""
    plane = RoadPlane.horizontal(0.0)
    X = np.array([450.0, -320.0, 0.0])
    h0, h1 = ground_homography(CAMS[0], plane), ground_homography(CAMS[1], plane)
    g, ok = pixel_to_ground(h0, CAMS[0].project(X)[None])
    assert ok[0]
    assert np.allclose(plane_to_pixel(h1, plane.to_plane(g)), CAMS[1].project(X)[None], atol=1e-4)


def test_online_road_model_fits_local_slope_and_reports_uncertainty():
    rng = np.random.default_rng(1)
    road = OnlineRoadModel(radius=3000.0, min_points=10, sigma_floor=5.0)
    assert road.local_plane(np.zeros(2)) is None
    xy = rng.uniform(-2000, 2000, (200, 2))
    z = 100.0 + 0.03 * xy[:, 0] - 0.01 * xy[:, 1] + rng.normal(0, 8.0, 200)
    for p in np.c_[xy, z]:
        road.add(p)
    plane = road.local_plane(np.array([500.0, 200.0]))
    assert plane.height_at(np.array([500.0, 200.0])) == pytest.approx(100 + 15 - 2, abs=4.0)
    assert np.allclose(plane.slope(), [0.03, -0.01], atol=0.005)
    assert 5.0 <= plane.sigma < 20.0
    # далеко от данных — неопределённее либо нет оценки
    far = road.local_plane(np.array([9000.0, 0.0]))
    assert far is None
    # буфер ограничен
    small = OnlineRoadModel(max_points=50)
    for p in np.c_[xy, z]:
        small.add(p)
    assert len(small) == 50
