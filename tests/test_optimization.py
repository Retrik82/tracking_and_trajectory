import math

import numpy as np
import pytest

from reconstruction.bspline import BSplineTrajectory, fit_initial_bspline
from reconstruction.camera import Camera
from reconstruction.metrics import (
    calculate_metrics,
    calculate_reprojection_errors,
    calculate_smoothness,
    compare_trajectories,
    position_error_stats,
)
from reconstruction.objective import (
    ObjectiveData,
    ObservationGeometry,
    TrajectoryObjective,
    project_point,
)
from reconstruction.optimizer import optimize_trajectory
from reconstruction.regularization import (
    acceleration_residuals,
    jerk_residuals,
    regularization_residuals,
    trapezoidal_weights,
)
from reconstruction.triangulation import triangulate_robust
from tests.conftest import load_target_positions, make_camera, requires_data

TIMES = np.arange(0, 201, dtype=float)


def true_path(t):
    t = np.asarray(t, dtype=float)
    return np.c_[1000 + 30 * t, 500 * np.sin(t / 40), 300 + 0.002 * t ** 2]


def look_at_camera(position, target, f=1500.0):
    d = np.asarray(target, float) - np.asarray(position, float)
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
    return Camera(position, 0.0, pitch, yaw, f, f, 960, 540, 1920, 1080)


@pytest.fixture(scope="module")
def cameras():
    center = true_path([100])[0]
    positions = [(-3000, -4000, 1500), (8000, -5000, 1800), (9000, 5000, 1200), (-2000, 6000, 2000)]
    return [look_at_camera(p, center) for p in positions]


@pytest.fixture(scope="module")
def true_spline():
    return fit_initial_bspline(TIMES, true_path(TIMES), knot_spacing=10, smoothing=0)


def observations(cameras, times, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t_list, idx_list, px_list = [], [], []
    for t in times:
        P = true_path([t])[0]
        for i, cam in enumerate(cameras):
            uv = cam.project(P)
            if np.all(np.isfinite(uv)) and 0 <= uv[0] < 1920 and 0 <= uv[1] < 1080:
                t_list.append(t)
                idx_list.append(i)
                px_list.append(uv + rng.normal(0, noise_px, 2))
    return ObjectiveData(
        times=np.array(t_list), camera_indices=np.array(idx_list),
        pixels=np.array(px_list), confidence=np.ones(len(t_list)),
    )


def perturbed(spline: BSplineTrajectory, scale=150.0, seed=1) -> BSplineTrajectory:
    rng = np.random.default_rng(seed)
    return BSplineTrajectory(spline.knots, spline.control_points + rng.normal(0, scale, spline.control_points.shape),
                             spline.degree, spline.t_start, spline.t_end)


# ---------------------------------------------------------------- regularization

def test_trapezoidal_weights_integrate_duration():
    t = np.array([2.0, 3.0, 5.0, 9.0])
    w = trapezoidal_weights(t)
    np.testing.assert_allclose(w, [0.5, 1.5, 3.0, 2.0])
    assert w.sum() == pytest.approx(t[-1] - t[0])


def test_regularization_uses_physical_time():
    # Раньше время нормализовалось в [0, 1]: при t_start = 100 сплайн вычислялся вне
    # интервала и возвращал NaN, а при t_start = 0 — неверный масштаб производной.
    t = np.arange(100, 301, dtype=float)
    spline = fit_initial_bspline(t, np.c_[t ** 2, 3 * t, np.zeros_like(t)], num_control_points=12, smoothing=0)
    samples = np.array([150.0, 200.0, 250.0])
    sqrt_w = np.sqrt(trapezoidal_weights(samples))[:, None]
    np.testing.assert_allclose(acceleration_residuals(spline, samples) / sqrt_w, [[2, 0, 0]] * 3, atol=1e-6)
    np.testing.assert_allclose(jerk_residuals(spline, samples) / sqrt_w, 0, atol=1e-6)


def test_regularization_residual_layout_and_weights():
    spline = fit_initial_bspline(TIMES, true_path(TIMES), knot_spacing=10)
    samples = np.linspace(0, 200, 11)
    r = regularization_residuals(spline, samples, lambda_acceleration=4.0, lambda_jerk=9.0)
    assert r.shape == (2 * 11 * 3,)
    np.testing.assert_allclose(r[:33], 2 * acceleration_residuals(spline, samples).ravel())
    np.testing.assert_allclose(r[33:], 3 * jerk_residuals(spline, samples).ravel())


def test_regularization_rejects_bad_input():
    linear = fit_initial_bspline(TIMES, true_path(TIMES), degree=1, num_control_points=5)
    with pytest.raises(ValueError):
        acceleration_residuals(linear, TIMES[:5])
    spline = fit_initial_bspline(TIMES, true_path(TIMES))
    with pytest.raises(ValueError):
        jerk_residuals(spline, TIMES[:5], weight=-1)


# ---------------------------------------------------------------- objective

def test_project_point_matches_camera_model(cameras):
    P = true_path([50, 120])
    for cam in cameras:
        for p in P:
            pixel, depth = project_point(cam, p)
            np.testing.assert_allclose(pixel, cam.project(p), atol=1e-9)
            assert depth == pytest.approx(cam.depth(p))
        geometry = ObservationGeometry.from_cameras([cam, cam])
        pixels, _ = geometry.project(P)
        np.testing.assert_allclose(pixels, cam.project(P), atol=1e-9)


def test_true_spline_has_zero_reprojection(cameras, true_spline):
    data = observations(cameras, TIMES[::5])
    objective = TrajectoryObjective(true_spline, cameras, data, lambda_acceleration=0, lambda_jerk=0)
    errors = calculate_reprojection_errors(true_spline, cameras, data)
    assert errors.max() < 0.05
    assert objective.error_components(true_spline.control_points.ravel())["depth_penalty"] == 0


def test_point_behind_camera_penalized_not_exploding(cameras, true_spline):
    data = observations(cameras, TIMES[::20])
    objective = TrajectoryObjective(true_spline, cameras, data, lambda_acceleration=0, lambda_jerk=0)
    behind = cameras[0].position - 1000 * cameras[0].forward
    cps = np.tile(behind, (len(true_spline.control_points), 1))
    r = objective.reprojection_residuals(BSplineTrajectory(true_spline.knots, cps, 3, 0, 200))
    count = len(data.times)
    cam0 = data.camera_indices == 0
    assert np.all(r[:2 * count].reshape(-1, 2)[cam0] == 0)
    assert np.all(r[2 * count:][cam0] > 0)
    assert np.all(np.isfinite(r))


def test_jac_sparsity_covers_true_jacobian(cameras, true_spline):
    data = observations(cameras, TIMES[::10])
    objective = TrajectoryObjective(true_spline, cameras, data, regularization_samples=15)
    x0 = true_spline.control_points.ravel() + 10.0
    base = objective.residuals(x0)
    pattern = objective.jac_sparsity().toarray().astype(bool)
    assert pattern.shape == (len(base), len(x0))
    for j in range(len(x0)):
        step = np.zeros_like(x0)
        step[j] = 1e-3
        changed = np.abs(objective.residuals(x0 + step) - base) > 1e-12
        assert not np.any(changed & ~pattern[:, j]), f"параметр {j}: зависимость вне структуры якобиана"


def test_validation_errors(cameras, true_spline):
    data = observations(cameras, TIMES[::20])
    bad = ObjectiveData(data.times, data.camera_indices + 10, data.pixels, data.confidence)
    with pytest.raises(ValueError, match="несуществующий"):
        TrajectoryObjective(true_spline, cameras, bad)
    bad = ObjectiveData(data.times + 1000, data.camera_indices, data.pixels, data.confidence)
    with pytest.raises(ValueError, match="вне интервала"):
        TrajectoryObjective(true_spline, cameras, bad)


def test_camera_provider_uses_pose_of_each_frame(true_spline):
    # Камера, которая поворачивается по yaw через ±180° (как Camera5).
    # Её поза должна браться на кадре наблюдения, а не одна на всё время.
    position = np.array([9000.0, 0.0, 900.0])

    def rotating(index, frame):
        target = true_path([frame])[0]
        cam = look_at_camera(position, target)
        return cam

    times = TIMES[::4]
    pixels = np.array([rotating(0, int(t)).project(true_path([t])[0]) for t in times])
    yaw = np.degrees([rotating(0, int(t)).yaw for t in times])
    assert yaw.max() > 170 and yaw.min() < -170          # проходим через ±180°
    data = ObjectiveData(times, np.zeros(len(times), int), pixels, np.ones(len(times)), frames=times.astype(int))

    errors = calculate_reprojection_errors(true_spline, rotating, data)
    assert errors.max() < 0.5
    frozen = rotating(0, 0)
    static_errors = calculate_reprojection_errors(true_spline, [frozen], ObjectiveData(
        times, np.zeros(len(times), int), pixels, np.ones(len(times))))
    assert np.nanmax(static_errors[np.isfinite(static_errors)]) > 100


# ---------------------------------------------------------------- optimizer

def test_optimizer_recovers_trajectory_from_noisy_pixels(cameras, true_spline):
    data = observations(cameras, TIMES, noise_px=1.0)
    start = perturbed(true_spline)
    result = optimize_trajectory(start, cameras, data, max_nfev=200)
    assert result.scipy_result.success
    assert result.final_error["total"] < result.initial_error["total"]

    before = position_error_stats(start, TIMES, true_path(TIMES), units_per_meter=100)
    after = position_error_stats(result.trajectory, TIMES, true_path(TIMES), units_per_meter=100)
    assert before["median"] > 1.0
    assert after["median"] < 0.1                       # м
    metrics = calculate_metrics(result.trajectory, cameras, data)
    assert metrics.reprojection_rmse < 2.0            # шум 1 px по каждой оси


def test_optimizer_huber_resists_outlier_detections(cameras, true_spline):
    data = observations(cameras, TIMES, noise_px=1.0, seed=3)
    rng = np.random.default_rng(4)
    bad = rng.choice(len(data.times), size=len(data.times) // 20, replace=False)
    data.pixels[bad] += rng.uniform(80, 200, (len(bad), 2))
    linear = optimize_trajectory(true_spline, cameras, data, loss="linear", max_nfev=200)
    huber = optimize_trajectory(true_spline, cameras, data, loss="huber", f_scale=3.0, max_nfev=200)
    err_linear = position_error_stats(linear.trajectory, TIMES, true_path(TIMES))["p95"]
    err_huber = position_error_stats(huber.trajectory, TIMES, true_path(TIMES))["p95"]
    assert err_huber < err_linear
    assert err_huber < 0.2


def test_optimizer_bridges_single_camera_segment(cameras, true_spline):
    # На кадрах 80–120 грузовик видит только одна камера: триангуляция там невозможна,
    # но сплайн, подогнанный по пикселям, остаётся ограничен этим лучом и соседями.
    data = observations(cameras, TIMES, noise_px=0.5, seed=5)
    keep = ~((data.times >= 80) & (data.times <= 120) & (data.camera_indices != 1))
    data = ObjectiveData(data.times[keep], data.camera_indices[keep], data.pixels[keep], data.confidence[keep])
    result = optimize_trajectory(perturbed(true_spline, 50), cameras, data, max_nfev=200)
    gap = np.arange(80, 121, dtype=float)
    assert position_error_stats(result.trajectory, gap, true_path(gap))["max"] < 1.0


# ---------------------------------------------------------------- metrics

def test_smoothness_of_quadratic():
    t = np.arange(100, 301, dtype=float)
    spline = fit_initial_bspline(t, np.c_[t ** 2, t, t], num_control_points=12, smoothing=0)
    acc_rms, jerk_rms = calculate_smoothness(spline)
    assert acc_rms == pytest.approx(2.0, rel=1e-6)
    assert jerk_rms == pytest.approx(0.0, abs=1e-6)


def test_metrics_per_camera_and_position(cameras, true_spline):
    data = observations(cameras, TIMES[::5], noise_px=1.0)
    metrics = calculate_metrics(true_spline, cameras, data)
    assert set(metrics.errors_by_camera) == set(np.unique(data.camera_indices).tolist())
    assert sum(v["count"] for v in metrics.errors_by_camera.values()) == len(data.times)

    target = true_path(TIMES)
    target[7] = np.nan                                 # испорченная ячейка эталона
    stats = position_error_stats(true_spline, TIMES, target)
    assert stats["count"] == len(TIMES) - 1
    assert stats["max"] < 0.01

    comparison = compare_trajectories(perturbed(true_spline), true_spline, cameras, data, TIMES, target)
    assert comparison["optimized"]["position_median_m"] < comparison["initial"]["position_median_m"]
    assert comparison["optimized"]["reprojection_rmse"] < comparison["initial"]["reprojection_rmse"]


# ---------------------------------------------------------------- реальные позы камер

@requires_data
def test_real_cameras_with_camera5_rotation():
    # Кадры 560–700: Camera5 начинает поворот (565) и проходит yaw ±180° (623/624).
    # Наблюдения — проекции точки из CSV цели с шумом 2 px, поза каждой камеры с её кадра.
    rng = np.random.default_rng(0)
    target = load_target_positions()
    frames = [f for f in range(560, 701) if f in target]

    t_obs, idx_obs, px_obs, tri_t, tri_p = [], [], [], [], []
    for f in frames:
        P = np.array(target[f])
        cams, dets = [], []
        for cam_id in range(1, 6):
            cam = make_camera(cam_id, f)
            uv = cam.project(P)
            if np.all(np.isfinite(uv)) and 0 <= uv[0] < 1920 and 0 <= uv[1] < 1080:
                uv = uv + rng.normal(0, 2.0, 2)
                t_obs.append(f)
                idx_obs.append(cam_id)
                px_obs.append(uv)
                cams.append(cam)
                dets.append(tuple(uv))
        if len(cams) >= 2:
            res = triangulate_robust(cams, dets)
            if res.ok:
                tri_t.append(f)
                tri_p.append(res.point)
    assert 5 in idx_obs

    initial = fit_initial_bspline(np.array(tri_t, float), np.array(tri_p), knot_spacing=10)
    data = ObjectiveData(np.array(t_obs, float), np.array(idx_obs), np.array(px_obs), np.ones(len(t_obs)),
                         frames=np.array(t_obs))
    inside = (data.times >= initial.t_start) & (data.times <= initial.t_end)
    data = ObjectiveData(data.times[inside], data.camera_indices[inside], data.pixels[inside],
                         data.confidence[inside], data.frames[inside])

    result = optimize_trajectory(initial, make_camera, data, max_nfev=100)
    eval_t = np.array([f for f in frames if initial.t_start <= f <= initial.t_end], float)
    gt = np.array([target[int(f)] for f in eval_t])
    comparison = compare_trajectories(initial, result.trajectory, make_camera, data, eval_t, gt)

    assert comparison["optimized"]["reprojection_rmse"] < 3.5          # шум 2 px по оси
    assert comparison["optimized"]["position_median_m"] < 0.3
    assert comparison["optimized"]["position_median_m"] <= comparison["initial"]["position_median_m"] * 1.05
