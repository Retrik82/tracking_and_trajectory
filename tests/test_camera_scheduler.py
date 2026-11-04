import math

import numpy as np

from reconstruction.cab_model import TruckDimensions
from reconstruction.camera import Camera
from reconstruction.camera_scheduler import (
    ACTIVE, PROBE, SLEEP, CameraScheduler, SchedulerSettings, TrackSnapshot, visibility_probability,
)

DIMS = TruckDimensions(1650.0, 255.0, 400.0).as_array()


def look_at(position, target, f=1500.0):
    d = np.asarray(target, float) - np.asarray(position, float)
    return Camera(position, 0.0, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]),
                  f, f, 960, 540, 1920, 1080)


# Камеры 1–2 смотрят на начало пути, камера 3 — на его конец (x ≈ 9000), камера 4 — вообще в сторону
CAMS = {1: look_at((-3000, -7000, 1500), (0, 0, 300)), 2: look_at((5000, -7000, 1800), (0, 0, 300)),
        3: look_at((9000, 7000, 1200), (9000, 0, 300)), 4: look_at((0, 9000, 1200), (0, 30000, 300))}


def snap(x, vx=50.0, std=100.0):
    return TrackSnapshot(np.array([x, 0.0, 300.0]), np.array([vx, 0.0, 0.0]), np.eye(3) * std ** 2,
                         np.array([1.0, 0.0, 0.0]), True, std / 100)


def test_visibility_probability_in_and_out_of_view():
    s = SchedulerSettings()
    p_in, _ = visibility_probability(CAMS[1], snap(0.0), s, None, DIMS)
    p_out, _ = visibility_probability(CAMS[4], snap(0.0), s, None, DIMS)
    assert p_in > 0.9 and p_out < 0.05
    # большая неопределённость — вероятность между 0 и 1
    p_unc, _ = visibility_probability(CAMS[3], snap(7000.0, std=3000.0), s, None, DIMS)
    assert 0.0 < p_unc < 1.0


def test_all_active_until_initialized_and_when_disabled():
    sched = CameraScheduler(SchedulerSettings(), CAMS, None, DIMS)
    plan = sched.plan(0, None, CAMS)
    assert plan.to_run == [1, 2, 3, 4] and set(plan.states.values()) == {ACTIVE}
    off = CameraScheduler(SchedulerSettings(enabled=False), CAMS, None, DIMS)
    assert off.plan(5, snap(0.0), CAMS).to_run == [1, 2, 3, 4]


def test_sleep_wake_hysteresis_and_probe_period():
    s = SchedulerSettings(hysteresis_frames=3, probe_period_frames=5, sleep_period_frames=0, min_active=2)
    sched = CameraScheduler(s, CAMS, None, DIMS)
    plans = []
    for f in range(0, 200):
        x = 50.0 * f                                     # объект едет от камер 1–2 к камере 3
        plan = sched.plan(f, snap(x), CAMS)
        sched.observe(f, {c: plan.p_now[c] > 0.5 for c in plan.to_run})
        plans.append(plan)
    states = np.array([[p.states[c] for c in (1, 2, 3, 4)] for p in plans])
    # камера 4 никогда не видит объект — засыпает и не просыпается
    assert np.all(states[10:, 3] == SLEEP)
    # камера 3 в начале спит или зондирует, к концу активна
    assert states[5, 2] in (PROBE, SLEEP) and states[-1, 2] == ACTIVE
    # камеры 1–2 в начале активны, к концу — не активны
    assert np.all(states[:5, :2] == ACTIVE) and np.all(states[-1, :2] != ACTIVE)
    # нет дребезга: число переходов у каждой камеры мало
    for col in range(4):
        assert (states[1:, col] != states[:-1, col]).sum() <= 4, col
    # PROBE: детекция не чаще раза в probe_period кадров
    probe_runs = [p.frame for p in plans if p.states[3] == PROBE and p.run[3] and not p.forced]
    assert all(b - a >= 5 for a, b in zip(probe_runs, probe_runs[1:]))
    # спящая камера не детектируется
    assert not any(p.run[4] for p in plans if p.states[4] == SLEEP)
    st = sched.stats()
    assert st["detection_fraction"] < 0.75 and st["mean_active"] < 3


def test_forced_activation_on_uncertainty_and_loss():
    s = SchedulerSettings(hysteresis_frames=2, probe_period_frames=10)
    sched = CameraScheduler(s, CAMS, None, DIMS)
    for f in range(10):
        plan = sched.plan(f, snap(0.0), CAMS)
        sched.observe(f, {c: c in (1, 2) for c in plan.to_run})
    assert sched.states[3] in (PROBE, SLEEP)
    lost = TrackSnapshot(np.array([0.0, 0, 300]), np.array([50.0, 0, 0]), np.eye(3) * 500 ** 2,
                         np.array([1.0, 0, 0]), True, 5.0, frames_since_measurement=3, cameras_used=0)
    plan = sched.plan(10, lost, CAMS)
    assert plan.forced and plan.run[1] and plan.run[2]
    # все кандидаты (объект может попасть в поле зрения) проверяются принудительно
    assert all(plan.run[c] for c in CAMS if plan.p_horizon[c] >= s.probe_threshold)
    # объект подъезжает к камере 3 с большой неопределённостью — она должна проснуться и проверяться
    near3 = TrackSnapshot(np.array([8000.0, 0, 300]), np.array([50.0, 0, 0]), np.eye(3) * 500 ** 2,
                          np.array([1.0, 0, 0]), True, 5.0, frames_since_measurement=3, cameras_used=0)
    plan = sched.plan(11, near3, CAMS)
    assert plan.forced and plan.run[3] and sched.states[3] in (PROBE, ACTIVE)


def test_occluded_camera_goes_to_probe_and_returns_on_detection():
    s = SchedulerSettings(occluded_patience_frames=5, hysteresis_frames=2)
    sched = CameraScheduler(s, CAMS, None, DIMS)
    for f in range(8):
        plan = sched.plan(f, snap(0.0), CAMS)
        sched.observe(f, {c: (c != 2) for c in plan.to_run})   # камера 2 видит геометрически, но не находит
    assert sched.states[2] == PROBE and sched.states[1] == ACTIVE
    plan = sched.plan(20, snap(0.0), CAMS)
    if plan.run[2]:
        sched.observe(20, {2: True})
    else:
        sched.observe(20, {2: True})
    assert sched.states[2] == ACTIVE
