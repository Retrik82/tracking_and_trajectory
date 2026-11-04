import math

import numpy as np
import pytest

from reconstruction.cab_model import BoxObservations, TruckDimensions, box_corners
from reconstruction.camera import Camera
from reconstruction.visibility import (
    INITIALIZING,
    MEASURED,
    PARTIAL,
    PREDICTED,
    CabTracker,
    TrackerSettings,
    detect_occluded_edges,
    run_tracker,
)

FRAMES = np.arange(0, 181)
DIMS = TruckDimensions(length=1650.0, width=255.0, height=400.0)
SPEED = 50.0                                   # ед. мира за кадр = 15 м/с


def path(t):
    t = np.asarray(t, dtype=float)
    return np.c_[SPEED * t, 1500 * np.sin(t / 120), 300 + 0 * t]


def velocity(t):
    t = np.asarray(t, dtype=float)
    return np.c_[np.full_like(t, SPEED), 1500 / 120 * np.cos(t / 120), 0 * t]


def look_at(position, target, f=1500.0):
    d = np.asarray(target, float) - np.asarray(position, float)
    return Camera(position, 0.0, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]),
                  f, f, 960, 540, 1920, 1080)


CAMERAS = [look_at(p, path([90])[0]) for p in [(-3000, -7000, 1500), (9000, -6000, 1800), (5000, 8000, 1200)]]


def provider(index, frame):
    return CAMERAS[index]


def synthetic(noise=0.5, seed=0, drop=None, shrink=None):
    """
    Рамки коробки с шумом. drop(cam, frame) -> True — детекции нет.
    shrink(cam, frame) -> пикселей, на которые правое ребро «съёживается» (заезд за препятствие).
    """
    rng = np.random.default_rng(seed)
    t, idx, edges, points = [], [], [], {}
    for f in FRAMES:
        corners = box_corners(path([f]), velocity([f]), DIMS.as_array())[0]
        for i, cam in enumerate(CAMERAS):
            if drop is not None and drop(i, f):
                continue
            uv = cam.project(corners)
            e = np.array([uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()]) + rng.normal(0, noise, 4)
            if shrink is not None:
                e[1] -= shrink(i, f)
            t.append(f)
            idx.append(i)
            edges.append(e)
            points[(i, int(f))] = tuple(cam.project(corners.mean(axis=0)))
    edges = np.array(edges)
    valid = np.c_[edges[:, 0] > 3, edges[:, 1] < 1917, edges[:, 2] > 3, edges[:, 3] < 1077]
    return BoxObservations(t, np.array(t, int), idx, edges, valid, np.ones(len(t))), points


def errors(result):
    pos = result.array("position")
    have = np.all(np.isfinite(pos), axis=1)
    return np.linalg.norm(pos - path(FRAMES), axis=1), have


def speed_jumps(result, settings):
    speed = result.array("speed_mps")
    return np.abs(np.diff(speed))[np.isfinite(np.diff(speed))] * settings.fps


# ---------------------------------------------------------------- закрытые рёбра

def test_detect_occluded_edges_one_sided_shrink_only():
    s = TrackerSettings()
    predicted = np.array([[500.0, 900.0, 300.0, 500.0]] * 4)
    observed = predicted.copy()
    observed[0, 1] -= 120                       # правое ребро ушло внутрь — закрыто
    observed[1, [0, 1]] += [-60, 60]            # рамка шире прогноза — не закрытие
    observed[2, 0] += 80
    observed[2, 1] -= 80                        # сжалась с обеих сторон — это не препятствие
    observed[3, 3] -= 70                        # низ закрыт (рельеф)
    valid = np.ones((4, 4), bool)
    occluded = detect_occluded_edges(predicted, observed, valid, s)
    assert occluded.tolist() == [[False, True, False, False], [False] * 4, [False] * 4, [False, False, False, True]]
    valid[0, 1] = False                         # обрезано краем кадра — уже учтено, не «закрытие»
    assert not detect_occluded_edges(predicted, observed, valid, s)[0].any()


# ---------------------------------------------------------------- сопровождение

def test_tracks_synthetic_truck():
    # На синтетике коробка точно совпадает с «грузовиком», поэтому СКО ребра — как у шума (по
    # умолчанию 20 px — с запасом на несовпадение коробки с реальным грузовиком)
    obs, points = synthetic()
    result = run_tracker(obs, provider, FRAMES, points, settings=TrackerSettings(edge_sigma_px=2.0))
    status = result.array("status")
    assert status[0] == INITIALIZING
    assert np.all(status[20:] == MEASURED)
    err, have = errors(result)
    assert np.all(have[20:])
    assert np.median(err[40:]) < 50.0                                   # см
    speed = result.array("speed_mps")[40:]
    true_speed = np.linalg.norm(velocity(FRAMES[40:]), axis=1) * 0.3
    assert np.median(np.abs(speed - true_speed)) < 0.5                  # м/с


def test_causal_past_estimates_ignore_future():
    obs, points = synthetic(seed=3)
    full = run_tracker(obs, provider, FRAMES, points)
    cut = 100
    past = obs.subset(obs.times <= cut)
    truncated = run_tracker(past, provider, FRAMES[:cut + 1], points)
    np.testing.assert_array_equal(truncated.array("position")[:cut + 1], full.array("position")[:cut + 1])
    np.testing.assert_array_equal(truncated.array("speed_mps")[:cut + 1], full.array("speed_mps")[:cut + 1])


def test_step_by_step_equals_batch():
    obs, points = synthetic(seed=4)
    batch = run_tracker(obs, provider, FRAMES[:60], points)
    tracker = CabTracker()
    for f in FRAMES[:60]:
        idx = np.nonzero(obs.times == f)[0]
        dets = [(int(obs.camera_indices[i]), obs.edges[i], obs.edge_valid[i], obs.confidence[i],
                 points[(int(obs.camera_indices[i]), int(f))], True) for i in idx]
        estimate = tracker.step(int(f), {int(obs.camera_indices[i]): CAMERAS[int(obs.camera_indices[i])] for i in idx},
                                dets)
        np.testing.assert_array_equal(estimate.position, batch.estimates[int(f)].position)


def test_full_loss_keeps_points_and_smooth_speed():
    lost = (80, 110)
    obs, points = synthetic(drop=lambda cam, f: lost[0] <= f <= lost[1])
    settings = TrackerSettings()
    result = run_tracker(obs, provider, FRAMES, points, settings=settings)
    status = result.array("status")
    assert np.all(status[lost[0]:lost[1] + 1] == PREDICTED)
    since = result.array("frames_since_measurement")
    assert since[lost[1]] == lost[1] - lost[0] + 1
    err, have = errors(result)
    assert np.all(have[20:])                                            # точка не пропадает
    assert err[lost[0]:lost[1] + 1].max() < 600                          # прогноз по вектору движения
    std = result.array("position_std")
    assert std[lost[1]] > std[lost[0] - 1]                              # неопределённость растёт
    assert speed_jumps(result, settings).max() <= settings.max_longitudinal_mps2 + settings.max_lateral_mps2


def test_occlusion_before_loss_is_handled():
    # Камера 0: с кадра 60 правое ребро «съёживается» на 12 px/кадр (заезд за дерево), с 80 — пропадает
    def shrink(cam, f):
        return 12.0 * (f - 60) if cam == 0 and 60 <= f < 80 else 0.0

    def drop(cam, f):
        return cam == 0 and f >= 80

    obs, points = synthetic(shrink=shrink, drop=drop, seed=1)
    handled = run_tracker(obs, provider, FRAMES, points)
    ignored = run_tracker(obs, provider, FRAMES, points, settings=TrackerSettings(occlusion_px=1e9))
    window = slice(60, 100)
    assert handled.array("occluded_edges")[window].sum() > 5
    err_handled, _ = errors(handled)
    err_ignored, _ = errors(ignored)
    assert err_handled[window].max() < err_ignored[window].max()
    assert err_handled[window].max() < 300


def test_reappearance_without_speed_jump():
    obs, points = synthetic(drop=lambda cam, f: cam != 2 and 70 <= f <= 120, seed=2)
    settings = TrackerSettings()
    result = run_tracker(obs, provider, FRAMES, points, settings=settings)
    jumps = speed_jumps(result, settings)
    assert jumps.max() <= settings.max_longitudinal_mps2 + settings.max_lateral_mps2


def test_partial_status_when_all_edges_occluded_or_clipped():
    obs, points = synthetic()
    obs.edge_valid[(obs.times == 150)] = [True, False, False, False]
    result = run_tracker(obs, provider, FRAMES, points)
    assert result.estimates[150].status == PARTIAL


def test_limits_clip_velocity_change():
    tracker = CabTracker()
    s = tracker.s
    prior = np.array([0.0, 0, 0, 50.0, 0, 0])
    posterior = np.array([0.0, 0, 0, 80.0, 30.0, 0])
    out = tracker._apply_limits(prior, posterior, prior[3:])
    long_max = s.max_longitudinal_mps2 * s.accel_to_units
    lat_max = s.max_lateral_mps2 * s.accel_to_units
    assert out[3] == pytest.approx(50.0 + long_max)
    assert out[4] == pytest.approx(lat_max)


def test_occlusion_decision_is_held_and_released_gradually():
    """Дребезг «закрыто/открыто» не должен менять набор измерений каждый кадр."""
    from reconstruction.visibility import CabTracker, TrackerSettings

    tracker = CabTracker(TrackerSettings(occlusion_hold_frames=3, occlusion_ramp_frames=4))
    det = [(1, None, None, 1.0, None, True)]
    flags = [True, False, True, False, False, False, False, False, False, False, False]
    weights = []
    for frame, flag in enumerate(flags):
        occluded = np.array([[flag, False, False, False]])
        weights.append(tracker._occlusion_hysteresis(frame, det, occluded)[0, 0])
    assert weights[:6] == [0.0] * 6                      # держится закрытым, несмотря на дребезг
    assert 0 < weights[6] < weights[7] < 1               # возвращается постепенно
    assert weights[-1] == 1.0
