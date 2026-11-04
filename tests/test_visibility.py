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
    tracker._confirmed = True                               # уверенный трек: пределы без ослабления
    tracker.P = np.eye(6)
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


# ---------------------------------------------------------------- модернизация: dt, ca, 3D-точка, курс

def test_variable_dt_matches_skipped_frames():
    obs, points = synthetic(seed=5)
    full = run_tracker(obs, provider, FRAMES, points)
    # тот же ряд, но каждый второй кадр после 60-го пропущен: dt = 2 кадра
    keep = np.r_[FRAMES[:60], FRAMES[60::2]]
    sub = obs.subset(np.isin(obs.times, keep))
    skipped = run_tracker(sub, provider, keep, points)
    err_full, _ = errors(full)
    pos = skipped.array("position")
    err_skip = np.linalg.norm(pos - path(keep), axis=1)
    assert np.all(skipped.array("dt_frames")[61:] == 2.0)
    assert np.nanmedian(err_skip[40:]) < 3 * max(np.nanmedian(err_full[40:]), 20.0)


def test_constant_acceleration_model_tracks():
    obs, points = synthetic(seed=6)
    settings = TrackerSettings(motion_model="ca", edge_sigma_px=2.0)
    result = run_tracker(obs, provider, FRAMES, points, settings=settings)
    err, have = errors(result)
    assert np.all(have[20:]) and np.median(err[40:]) < 60.0
    acc = result.array("accel_magnitude_mps2")
    assert np.all(np.isfinite(acc[20:]))


def test_point_fusion_improves_or_keeps_accuracy_and_reports_quality():
    from reconstruction.fusion import FusionSettings

    obs, points = synthetic(seed=7, noise=3.0)
    base = run_tracker(obs, provider, FRAMES, points)
    fused = run_tracker(obs, provider, FRAMES, points, fusion=FusionSettings(mode="model_corrected", sigma_px=3.0))
    err_b, _ = errors(base)
    err_f, _ = errors(fused)
    assert np.median(err_f[40:]) <= np.median(err_b[40:]) * 1.2
    used = fused.array("point_used")
    assert used[40:].mean() > 0.9
    assert set(fused.array("quality")[40:]) <= {"point+edges", "point", "edges"}
    nis = fused.array("point_nis")
    assert np.nanmedian(nis[40:]) < 16


def test_point_outlier_is_gated():
    from reconstruction.fusion import FusionSettings

    obs, points = synthetic(seed=8)
    bad = dict(points)
    ghost = path([100])[0] + [0.0, 600.0, 0.0]                  # согласованный ложный объект в 6 м сбоку
    for cam in range(3):
        bad[(cam, 100)] = tuple(CAMERAS[cam].project(ghost))
    fused = run_tracker(obs, provider, FRAMES, bad, fusion=FusionSettings(mode="centroid"))
    e = fused.estimates[100]
    assert not e.point_used and e.point_nis > 16
    assert fused.estimates[99].point_used and fused.estimates[101].point_used


def test_heading_rate_limited_and_held_at_low_speed():
    tracker = CabTracker(TrackerSettings(heading_rate_limit_dps=30.0))
    tracker.heading = 0.0
    tracker.heading_dir = np.array([1.0, 0.0, 0.0])
    fast = np.array([0, 0, 0, 0.0, 50.0, 0.0])              # 15 м/с строго вдоль +Y (90°)
    tracker._update_heading(fast, 1.0)
    assert abs(np.degrees(tracker.heading) - 1.0) < 1e-6    # 30°/с = 1°/кадр (v/R = 107°/с не ограничивает)
    tracker.heading = 0.0
    tracker._v_smooth = None
    crawl = np.array([0, 0, 0, 0.0, 20.0 / 3.0, 0.0])       # 2 м/с: предел v/R = 2/8 рад/с = 14.3°/с = 0.48°/кадр
    tracker._update_heading(crawl, 1.0)
    assert abs(np.degrees(tracker.heading) - np.degrees(2.0 / 8.0 / 30.0)) < 1e-6
    slow = np.array([0, 0, 0, 0.0, 1.0, 0.0])               # 0.3 м/с — ниже порога, курс держится
    before = tracker.heading
    tracker._update_heading(slow, 1.0)
    assert tracker.heading == before


def test_false_detection_is_gated_when_track_confident():
    obs, points = synthetic(seed=9)
    bad = obs.edges.copy()
    rows = np.nonzero((obs.times >= 100) & (obs.times < 110) & (obs.camera_indices == 0))[0]
    bad[rows, 0] += 500                                     # ложная рамка далеко от прогноза на камере 0
    bad[rows, 1] += 500
    wrong = BoxObservations(obs.times, obs.frames, obs.camera_indices, bad, obs.edge_valid, obs.confidence)
    gated = run_tracker(wrong, provider, FRAMES, points)
    ungated = run_tracker(wrong, provider, FRAMES, points, settings=TrackerSettings(gate_iou=-1.0))
    rej = [e.rejected_cameras for e in gated.estimates]
    assert all(0 in r for r in rej[100:110])
    e_g, _ = errors(gated)
    e_u, _ = errors(ungated)
    assert e_g[100:115].max() < e_u[100:115].max()


def test_mono_initialization_with_single_camera():
    obs, points = synthetic(seed=10, drop=lambda cam, f: cam != 1)      # видит только камера 1
    result = run_tracker(obs, provider, FRAMES, points, settings=TrackerSettings(mono_init=True))
    status = result.array("status")
    assert np.any(status == MEASURED)
    assert all(q == "mono" for q in result.array("quality")[np.nonzero(status == MEASURED)[0]])
    err, have = errors(result)
    assert have[60:].all() and np.nanmedian(err[60:]) < 1500          # грубо (дальность по высоте рамки), но трек есть
    off = run_tracker(obs, provider, FRAMES, points)                   # по умолчанию выключено
    assert np.all(off.array("status") == INITIALIZING)


def test_stereo_reinitializes_unconfirmed_mono_track():
    obs, points = synthetic(seed=11, drop=lambda cam, f: cam != 1 and f < 60)   # до кадра 60 — одна камера
    result = run_tracker(obs, provider, FRAMES, points, settings=TrackerSettings(mono_init=True))
    q = result.array("quality")
    assert "mono" in set(q[20:60]) and "mono" not in set(q[80:])
    err, _ = errors(result)
    assert np.nanmedian(err[100:]) < 100


def test_diverged_track_reinitializes_from_stereo():
    from reconstruction.fusion import FusionSettings

    obs, points = synthetic(seed=12)
    settings = TrackerSettings(edge_sigma_px=2.0)
    tracker = CabTracker(settings, DIMS, None, FusionSettings(mode="centroid", sigma_px=2.0))
    by_frame = {}
    for i, t in enumerate(obs.times.astype(int)):
        by_frame.setdefault(t, []).append(i)
    err = []
    for f in FRAMES:
        idx = by_frame.get(int(f), [])
        dets = [(int(obs.camera_indices[i]), obs.edges[i], obs.edge_valid[i], obs.confidence[i],
                 points[(int(obs.camera_indices[i]), int(f))], True) for i in idx]
        cams = {int(obs.camera_indices[i]): CAMERAS[int(obs.camera_indices[i])] for i in idx}
        if f == 60:                                        # сбить трек: унести состояние на 60 м вбок
            tracker.state[:3] += [0.0, 6000.0, 0.0]
            tracker.P[:3, :3] = np.eye(3) * 100.0 ** 2
        e = tracker.step(int(f), cams, dets)
        err.append(np.linalg.norm(e.position - path([f])[0]) if np.all(np.isfinite(e.position)) else np.nan)
    err = np.array(err)
    assert err[61] > 3000                                   # сразу после сбоя трек далеко
    assert np.nanmedian(err[100:]) < 100                    # после переинициализации — снова точен


def test_camera_seen_on_every_tick_is_not_reappearing_when_ticks_skip_frames():
    """Тики через 3 кадра (реальный темп): камера, видимая на каждом тике, не «появляется» заново."""
    data, points = synthetic()
    tracker = CabTracker(TrackerSettings(), DIMS)
    by_frame = {}
    for i, t in enumerate(data.times.astype(int)):
        by_frame.setdefault(t, []).append(i)
    for f in range(0, 60, 3):
        dets = [(int(data.camera_indices[i]), data.edges[i], data.edge_valid[i], 1.0,
                 points.get((int(data.camera_indices[i]), f)), True) for i in by_frame[f]]
        tracker.step(f, {d[0]: provider(d[0], f) for d in dets}, dets, dt_frames=3.0)
    assert all(v == 0 for v in tracker._appeared.values())


def test_size_cut_edge_is_detected_even_when_prediction_lags():
    """Рамка вдвое уже модели: срезано ребро, отклонившееся сильнее (отставание прогноза сдвигает оба ребра)."""
    tracker = CabTracker(TrackerSettings(occlusion_size_ratio=0.8), DIMS)
    predicted = np.array([[100.0, 400.0, 200.0, 300.0]])
    lag = 30.0                                            # прогноз отстаёт: обе стороны сдвинуты на 30 px
    observed = np.array([[100.0 + lag + 150.0, 400.0 + lag, 200.0, 300.0]])   # левая часть закрыта
    cut = tracker._size_cut_edges(predicted, observed, np.ones((1, 4), bool))
    assert cut.tolist() == [[True, False, False, False]]
    whole = np.array([[100.0 + lag, 400.0 + lag, 200.0, 300.0]])
    assert not tracker._size_cut_edges(predicted, whole, np.ones((1, 4), bool)).any()


def test_output_heading_respects_turn_rate_and_yaw_acceleration_limits():
    """Итоговый курс не дёргается: шаг ≤ (v/R_min + пол)·dt и изменение скорости поворота ≤ предела."""
    data, points = synthetic()
    s = TrackerSettings(heading_output_rate_limit=True, heading_accel_limit_dps2=120.0, min_turn_radius_m=8.0)
    tracker = CabTracker(s, DIMS)
    by_frame = {}
    for i, t in enumerate(data.times.astype(int)):
        by_frame.setdefault(t, []).append(i)
    headings = []
    for f in FRAMES:
        dets = [(int(data.camera_indices[i]), data.edges[i], data.edge_valid[i], 1.0,
                 points.get((int(data.camera_indices[i]), int(f))), True) for i in by_frame.get(int(f), [])]
        e = tracker.step(int(f), {d[0]: provider(d[0], int(f)) for d in dets}, dets)
        if np.isfinite(e.heading):
            if headings and np.random.default_rng(int(f)).random() < 0.3:
                tracker.heading += math.radians(10)            # имитация рывка поправки курса
            headings.append(e.heading)
    step = np.abs(np.diff(np.unwrap(headings)))
    max_rate = (SPEED * 30 / 100) / 8.0 + math.radians(5.0)    # рад/с
    assert step.max() <= max_rate / 30 + 1e-9
    assert np.abs(np.diff(np.diff(np.unwrap(headings)))).max() <= math.radians(120.0) / 30 ** 2 * 2 + 1e-6
