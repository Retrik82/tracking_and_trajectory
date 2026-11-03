import math

import numpy as np
import pytest

from detection.schema import Detection
from reconstruction.bspline import BSplineTrajectory, fit_initial_bspline
from reconstruction.cab_model import (
    BoxObservations,
    CabObjective,
    TruckDimensions,
    box_corners,
    fit_cab_trajectory,
    initial_cab_trajectory,
    offset_point_trajectory,
    project_box,
    truck_axes,
)
from reconstruction.camera import Camera
from reconstruction.metrics import position_error_stats
from reconstruction.objective import ObjectiveData, ObservationGeometry, observation_cameras

TIMES = np.arange(0, 121, dtype=float)
DIMS = TruckDimensions(length=1700.0, width=255.0, height=400.0)


def cab_path(t):
    t = np.asarray(t, dtype=float)
    return np.c_[30 * t, 1500 * np.sin(t / 60), 300 + 0 * t]


def look_at(position, target, f=1500.0):
    d = np.asarray(target, float) - np.asarray(position, float)
    return Camera(position, 0.0, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]),
                  f, f, 960, 540, 1920, 1080)


@pytest.fixture(scope="module")
def cameras():
    mid = cab_path([60])[0]
    return [look_at(p, mid) for p in [(-4000, -6000, 1500), (7000, -7000, 1800), (6000, 7000, 1200)]]


@pytest.fixture(scope="module")
def true_spline():
    return fit_initial_bspline(TIMES, cab_path(TIMES), knot_spacing=10, smoothing=0)


def make_observations(cameras, spline, cams=(0, 1, 2), noise=0.0, seed=0, times=TIMES):
    rng = np.random.default_rng(seed)
    t, idx, edges = [], [], []
    for ti in times:
        p = spline.evaluate(np.array([ti]))
        corners = box_corners(p, spline.derivative().evaluate(np.array([ti])), DIMS.as_array())[0]
        for i in cams:
            uv = cameras[i].project(corners)
            t.append(ti)
            idx.append(i)
            edges.append([uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()] + rng.normal(0, noise, 4))
    edges = np.array(edges)
    valid = np.c_[edges[:, 0] > 3, edges[:, 1] < 1917, edges[:, 2] > 3, edges[:, 3] < 1077]
    return BoxObservations(t, np.array(t, int), idx, edges, valid, np.ones(len(t)))


def shifted(spline, offset):
    return BSplineTrajectory(spline.knots, spline.control_points + offset, spline.degree, spline.t_start, spline.t_end)


def test_truck_axes_orthonormal_and_up():
    forward, side, up = truck_axes(np.array([[3.0, 4.0, 0.0], [-1.0, 0.0, 0.2], [0.0, -2.0, -0.5]]))
    for f, s, u in zip(forward, side, up):
        M = np.vstack([f, s, u])
        np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
        assert u[2] > 0 and s[2] == 0


def test_box_corners_geometry():
    corners = box_corners(np.array([[100.0, 200.0, 300.0]]), np.array([[1.0, 0.0, 0.0]]), DIMS.as_array())[0]
    np.testing.assert_allclose(corners.min(0), [100 - 1700, 200 - 127.5, 300])
    np.testing.assert_allclose(corners.max(0), [100, 200 + 127.5, 700])


def test_project_box_matches_camera(cameras, true_spline):
    t = np.array([10.0, 60.0])
    corners = box_corners(true_spline.evaluate(t), true_spline.derivative().evaluate(t), DIMS.as_array())
    data = ObjectiveData(t, np.array([0, 1]), np.zeros((2, 2)), np.ones(2))
    geometry = ObservationGeometry.from_cameras(observation_cameras(cameras, data))
    box = project_box(geometry, corners)
    for i, cam in enumerate((cameras[0], cameras[1])):
        uv = cam.project(corners[i])
        np.testing.assert_allclose(box[i], [uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()])


def test_true_parameters_give_zero_edge_error(cameras, true_spline):
    obs = make_observations(cameras, true_spline)
    objective = CabObjective(true_spline, cameras, obs, lambda_acceleration=0, lambda_jerk=0)
    theta = np.r_[true_spline.control_points.ravel(), DIMS.as_array()]
    r = objective.residuals(theta)[:4 * len(obs.times)]
    assert np.abs(r).max() < 1e-6


def test_jac_sparsity_covers_dependencies(cameras, true_spline):
    obs = make_observations(cameras, true_spline, times=TIMES[::15])
    objective = CabObjective(true_spline, cameras, obs, regularization_samples=10)
    theta = np.r_[true_spline.control_points.ravel() + 5.0, DIMS.as_array()]
    base = objective.residuals(theta)
    pattern = objective.jac_sparsity().toarray().astype(bool)
    for j in range(len(theta)):
        step = np.zeros_like(theta)
        step[j] = 1e-2
        changed = np.abs(objective.residuals(theta + step) - base) > 1e-10
        assert not np.any(changed & ~pattern[:, j]), j


def test_fit_recovers_cab_and_dimensions(cameras, true_spline):
    obs = make_observations(cameras, true_spline, noise=1.0)
    start = shifted(true_spline, np.array([300.0, -200.0, 100.0]))
    fit = fit_cab_trajectory(start, cameras, obs, dimensions=TruckDimensions(1500, 300, 350),
                             dimension_bounds=None, max_nfev=500)
    err = position_error_stats(fit.trajectory, TIMES, cab_path(TIMES))
    assert err["median"] < 0.15                                          # м
    np.testing.assert_allclose(fit.dimensions.as_array(), DIMS.as_array(), rtol=0.05)
    assert np.nanmedian(fit.edge_errors_px) < 2.0


def test_single_camera_segment_between_multi_camera(cameras, true_spline):
    # Кадры 40–80 видит только одна камера: между участками с тремя камерами
    # рамки одной камеры удерживают траекторию
    obs = make_observations(cameras, true_spline, noise=0.5)
    obs = obs.subset(~((obs.times >= 40) & (obs.times <= 80) & (obs.camera_indices != 1)))
    start = shifted(true_spline, np.array([300.0, 200.0, 0.0]))
    gap = np.arange(40, 81, dtype=float)
    for fit_dimensions in (True, False):
        fit = fit_cab_trajectory(start, cameras, obs, dimensions=DIMS, fit_dimensions=fit_dimensions,
                                 dimension_bounds=None, max_nfev=1000)
        assert position_error_stats(fit.trajectory, gap, cab_path(gap))["max"] < 0.3, fit_dimensions
    assert fit.dimensions.as_array().tolist() == DIMS.as_array().tolist()


def test_clipped_edges_are_ignored(cameras, true_spline):
    obs = make_observations(cameras, true_spline)
    obs.edges[::3, 1] += 400                       # «обрезанное» ребро с мусорным значением
    obs.edge_valid[::3, 1] = False
    start = shifted(true_spline, np.array([200.0, 0.0, 0.0]))
    fit = fit_cab_trajectory(start, cameras, obs, dimensions=DIMS, fit_dimensions=False, max_nfev=300)
    assert position_error_stats(fit.trajectory, TIMES, cab_path(TIMES))["median"] < 0.05
    assert np.all(np.isnan(fit.edge_errors_px[::3, 1]))


def test_rotating_camera_provider(true_spline):
    position = np.array([7000.0, 5000.0, 900.0])

    def provider(index, frame):
        return look_at(position, cab_path([frame])[0])

    obs_cams = [provider(0, int(t)) for t in TIMES]
    t, edges = [], []
    for ti, cam in zip(TIMES, obs_cams):
        corners = box_corners(true_spline.evaluate(np.array([ti])), true_spline.derivative().evaluate(np.array([ti])),
                              DIMS.as_array())[0]
        uv = cam.project(corners)
        t.append(ti)
        edges.append([uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()])
    obs = BoxObservations(t, np.array(t, int), np.zeros(len(t), int), edges, np.ones((len(t), 4), bool), np.ones(len(t)))
    objective = CabObjective(true_spline, provider, obs, lambda_acceleration=0, lambda_jerk=0)
    r = objective.residuals(np.r_[true_spline.control_points.ravel(), DIMS.as_array()])[:4 * len(t)]
    assert np.abs(r).max() < 1e-6


def test_from_detections_and_initial_guess(true_spline):
    d = Detection(camera=2, frame=10, confidence=0.9, x1=0.0, y1=100.0, x2=500.0, y2=1080.0,
                  point_u=250.0, point_v=500.0, point_source="mask_centroid")
    obs = BoxObservations.from_detections([d], frame_offset=1)
    assert obs.frames[0] == 11 and obs.times[0] == 10
    assert obs.edge_valid[0].tolist() == [False, True, True, False]

    center = BSplineTrajectory(true_spline.knots, true_spline.control_points, 3, 0, 120)
    init = initial_cab_trajectory(center, DIMS, knot_spacing=20)
    forward, _, up = truck_axes(center.derivative().evaluate(np.array([60.0])))
    expected = center.evaluate(np.array([60.0])) + DIMS.length / 2 * forward - DIMS.height / 2 * up
    np.testing.assert_allclose(init.evaluate(np.array([60.0])), expected, atol=30)


def test_bad_observations_rejected():
    with pytest.raises(ValueError):
        BoxObservations([1.0], [1], [0, 1], [[0, 1, 0, 1]], [[True] * 4], [1.0])
    with pytest.raises(ValueError):
        BoxObservations([], [], [], np.zeros((0, 4)), np.zeros((0, 4), bool), [])


def test_cab_center_offset_in_truck_axes(true_spline):
    center = offset_point_trajectory(true_spline, back=120.0, up=200.0)
    t = np.array([5.0, 60.0, 115.0])
    forward, _, up = truck_axes(true_spline.derivative().evaluate(t))
    expected = true_spline.evaluate(t) - 120.0 * forward + 200.0 * up
    np.testing.assert_allclose(center.evaluate(t), expected, atol=1.0)
    np.testing.assert_allclose(center.knots, true_spline.knots)
    # Нулевое смещение — та же траектория
    np.testing.assert_allclose(offset_point_trajectory(true_spline, 0, 0).evaluate(t), true_spline.evaluate(t), atol=1e-6)
