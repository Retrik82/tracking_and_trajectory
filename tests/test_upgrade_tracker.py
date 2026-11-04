"""Доработка трекера: дальность по габаритам, передача между камерами, неопределённость, сценарии."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pipeline import scenarios as S
from pipeline.config import StreamConfig
from pipeline.variants import VARIANTS
from reconstruction.camera import Camera
from reconstruction.objective import ObservationGeometry
from reconstruction.visibility import CabTracker, TrackerSettings, monocular_range
from reconstruction.cab_model import TruckDimensions

DIMS = TruckDimensions(1650.0, 255.0, 400.0)


def look_at(position, target, f=1600.0):
    d = np.asarray(target, float) - np.asarray(position, float)
    return Camera(position, 0.0, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]),
                  f, f, 960, 540, 1920, 1080)


def box(tracker, cam, center, heading):
    d = np.array([math.cos(heading), math.sin(heading), 0.0])
    return tracker._measure(np.asarray(center, float), d, ObservationGeometry.from_cameras([cam]))[0]


# ---------------------------------------------------------------- дальность по габаритам

@pytest.mark.parametrize("heading_deg", [0.0, 60.0, 135.0])
def test_monocular_range_from_known_dimensions(heading_deg):
    tracker = CabTracker(TrackerSettings(), DIMS)
    cam = look_at((-6000, -3000, 1500), (0, 0, 200))
    truth = np.array([0.0, 0.0, 200.0])
    ray = (truth - cam.position) / np.linalg.norm(truth - cam.position)
    r_true = float(np.linalg.norm(truth - cam.position))
    prior = cam.position + ray * r_true * 0.85                     # прогноз на 15 % ближе по тому же лучу
    h = math.radians(heading_deg)
    r = monocular_range(r_true * 0.85, box(tracker, cam, prior, h), box(tracker, cam, truth, h))
    assert r == pytest.approx(r_true, rel=0.03)


def test_monocular_range_rejects_occluded_clipped_and_tiny_boxes():
    tracker = CabTracker(TrackerSettings(), DIMS)
    cam = look_at((-6000, -3000, 1500), (0, 0, 200))
    center = np.array([0.0, 0.0, 200.0])
    r = float(np.linalg.norm(center - cam.position))
    pred = box(tracker, cam, center, 0.3)
    occluded = pred.copy()
    occluded[1] -= 0.35 * (pred[1] - pred[0])                      # правая часть закрыта
    assert np.isnan(monocular_range(r, pred, occluded))
    clipped = pred.copy()
    clipped[3] = clipped[2] + 0.5 * (pred[3] - pred[2])            # обрезано низом кадра
    assert np.isnan(monocular_range(r, pred, clipped))
    tiny = np.array([100.0, 105.0, 100.0, 104.0])
    assert np.isnan(monocular_range(r, pred, tiny))


# ---------------------------------------------------------------- сценарии: передача, восстановление, неопределённость

def _sc(name):
    return {s.name: s for s in S.scenarios()}[name]


def test_handoff_rejects_neighbouring_object_on_reappearing_camera():
    base = S.run_scenario(_sc("чужой объект при появлении"), "A")
    rec = S.run_scenario(_sc("чужой объект при появлении"), "REC")
    assert rec["rejected_detections"] >= 20
    assert rec["error_m"]["p95"] < 1.0 < base["error_m"]["p95"]
    assert rec["initializations"] == 1                         # трек не пересоздаётся при смене камер


def test_disappearing_cameras_and_full_loss_keep_one_track_without_jump():
    two = S.run_scenario(_sc("нет двух камер"), "REC")
    assert two["initializations"] == 1 and two["error_m"]["max"] < 1.5
    loss = S.run_scenario(_sc("потеря и восстановление"), "REC")
    assert loss["initializations"] == 1
    assert loss["max_position_jump_m"] < 2.0                  # возврат к измерениям без резкого скачка
    assert loss["levels"].get("predicted", 0) >= 25          # прогноз помечен как прогноз, не как измерение


def test_stereo_loss_upgrade_keeps_track_and_calibrated_uncertainty():
    base = S.run_scenario(_sc("потеря стерео"), "A")
    rec = S.run_scenario(_sc("потеря стерео"), "REC")
    assert rec["error_m"]["p95"] < 6.0 < base["error_m"]["p95"]
    assert rec["coverage_95"] > 0.85
    assert rec["levels"].get("single", 0) >= 100              # однокамерные кадры помечены
    assert rec["initializations"] == 1


def test_slow_turn_heading_from_image():
    base = S.run_scenario(_sc("медленный поворот"), "A")
    rec = S.run_scenario(_sc("медленный поворот"), "A+O")
    assert rec["heading_error_deg"]["median"] < 3.0 < base["heading_error_deg"]["median"]
    assert rec["error_m"]["p95"] < 1.0


def test_partial_occlusion_and_wrong_mask_do_not_turn_the_model():
    occ = S.run_scenario(_sc("частичное перекрытие"), "REC")
    assert occ["heading_error_deg"]["p95"] < 6.0
    wrong = S.run_scenario(_sc("неверная ориентация маски"), "REC")
    assert wrong["heading_error_deg"]["p95"] < 6.0


# ---------------------------------------------------------------- неопределённость на уровне трекера

def _run_tracker(settings, drop_after=None, frames=160, jump=None):
    """Коробка едет по прямой, 3 камеры; drop_after — с этого кадра видна одна камера; jump — скачок рамок."""
    cams = [look_at((-3000, -7000, 1500), (3000, 0, 0)), look_at((9000, -6000, 1800), (3000, 0, 0)),
            look_at((5000, 8000, 1200), (3000, 0, 0))]
    tracker = CabTracker(settings, DIMS)
    out = []
    for f in range(frames):
        center = np.array([40.0 * f, 0.0, 200.0])
        dets = []
        for i, cam in enumerate(cams):
            if drop_after is not None and f >= drop_after and i != 2:
                continue
            c = center + (np.array([600.0, 0.0, 0.0]) if jump is not None and f == jump else 0.0)
            e = box(tracker, cam, c, 0.0)
            dets.append((i, e, np.ones(4, bool), 0.9, tuple(cam.project(center)), True))
        out.append(tracker.step(f, {i: cams[i] for i in range(3)}, dets))
    return tracker, out


def test_covariance_grows_without_stereo_and_along_camera_ray():
    s = TrackerSettings(no_stereo_q_growth_per_s=0.5, depth_sigma_growth_mps=1.0)
    _, est = _run_tracker(s, drop_after=80)
    std = np.array([e.position_std for e in est])
    assert est[100].observation_level == "single" and est[60].observation_level == "multi"
    assert std[150] > 2 * std[70]
    assert np.all(np.diff(std[85:150]) > -1e-6)                    # без стерео неопределённость не убывает
    # рост — вдоль луча единственной камеры
    cam2_pos = np.array([5000.0, 8000.0, 1200.0])
    ray = est[150].box_center - cam2_pos
    ray /= np.linalg.norm(ray)
    w, v = np.linalg.eigh(est[150].covariance)
    assert abs(v[:, -1] @ ray) > 0.9
    _, base = _run_tracker(TrackerSettings(), drop_after=80)
    assert est[150].position_std > base[150].position_std


def test_uncertainty_returns_after_stereo_recovers():
    s = TrackerSettings(no_stereo_q_growth_per_s=0.5, depth_sigma_growth_mps=1.0)
    cams_back = 120                                                # стерео пропадает на 60 и возвращается на 120
    tracker = CabTracker(s, DIMS)
    cams = [look_at((-3000, -7000, 1500), (3000, 0, 0)), look_at((9000, -6000, 1800), (3000, 0, 0)),
            look_at((5000, 8000, 1200), (3000, 0, 0))]
    stds = []
    for f in range(200):
        center = np.array([40.0 * f, 0.0, 200.0])
        dets = [(i, box(tracker, c, center, 0.0), np.ones(4, bool), 0.9, tuple(c.project(center)), True)
                for i, c in enumerate(cams) if not (60 <= f < cams_back and i != 2)]
        stds.append(tracker.step(f, {i: cams[i] for i in range(3)}, dets).position_std)
    stds = np.array(stds)
    assert stds[cams_back - 1] > 2 * stds[cams_back + 30]
    assert stds[cams_back + 30] < 1.5 * stds[55]


def test_clipping_does_not_shrink_covariance():
    on = TrackerSettings(clip_covariance=True, max_correction_per_frame=50.0)
    off = TrackerSettings(clip_covariance=False, max_correction_per_frame=50.0)
    _, e_on = _run_tracker(on, jump=60, frames=62)
    _, e_off = _run_tracker(off, jump=60, frames=62)
    assert e_on[60].clip_correction > 0
    diff = e_on[60].covariance - e_off[60].covariance
    assert np.linalg.eigvalsh(diff).min() > -1e-6                  # P_clip ⪰ P_без коррекции
    assert np.trace(diff) > 0


def test_covariances_symmetric_psd_and_include_heading_and_model_terms():
    s = TrackerSettings(heading_sigma_floor_deg=5.0, model_length_sigma=60.0, clip_covariance=True)
    _, est = _run_tracker(s)
    _, base = _run_tracker(TrackerSettings())
    for e in est[20:]:
        for C in (e.covariance, e.reference_covariance):
            assert np.allclose(C, C.T, atol=1e-9)
            assert np.linalg.eigvalsh(C).min() > -1e-9
    e, b = est[100], base[100]
    # рычаг точки вывода × СКО курса (≥ 5°) и половина погрешности длины — вдоль оси
    extra = e.reference_covariance - e.covariance
    lever = np.linalg.norm(e.position[:2] - e.box_center[:2])
    assert extra[1, 1] >= (lever * math.radians(5.0)) ** 2 * 0.99
    assert extra[0, 0] >= (60.0 / 2) ** 2 * 0.99
    assert e.heading_std >= math.radians(5.0) * 0.999 > b.heading_std


def test_expected_view_matches_observation():
    tracker, est = _run_tracker(TrackerSettings(), frames=60)
    cam = look_at((-3000, -7000, 1500), (3000, 0, 0))
    view = tracker.expected_view(cam)
    truth = box(tracker, cam, np.array([40.0 * 59, 0.0, 200.0]), 0.0)
    c_true = np.array([(truth[0] + truth[1]) / 2, (truth[2] + truth[3]) / 2])
    assert view["visible"]
    assert np.linalg.norm(view["center_px"] - c_true) < 10.0
    assert np.all(np.linalg.eigvalsh(view["center_cov"]) > 0)


def test_strict_initialization_rejects_parallel_rays():
    """Две почти соосные камеры (угол лучей < порога) не подтверждают 3D-точку: трек не стартует."""
    cams = [look_at((-8000, -100, 1500), (0, 0, 0)), look_at((-8000, 100, 1500), (0, 0, 0))]
    strict = CabTracker(TrackerSettings(init_min_ray_angle_deg=10.0), DIMS)
    loose = CabTracker(TrackerSettings(), DIMS)
    for f in range(40):
        center = np.array([60.0 * f - 1000, 0.0, 200.0])
        for tr in (strict, loose):
            dets = [(i, box(tr, c, center, 0.0), np.ones(4, bool), 0.9, tuple(c.project(center)), True)
                    for i, c in enumerate(cams)]
            tr.step(f, {0: cams[0], 1: cams[1]}, dets)
    assert strict.state is None and strict.init_rejected > 0
    assert loose.state is not None


# ---------------------------------------------------------------- конфигурация

def test_default_config_keeps_baseline_and_recommended_loads():
    cfg = StreamConfig()
    assert not cfg.orientation.enabled and not cfg.homography.enabled and not cfg.mono_depth.enabled
    assert cfg.road.mode == "none" and not cfg.uncertainty.clip_covariance
    rec = StreamConfig.load(Path("configs/upgrade_recommended.json"))
    assert rec.orientation.enabled and rec.road.mode == "online" and not rec.mono_depth.enabled
    cfg2 = StreamConfig.load(None, ["orientation.enabled=true", "uncertainty.model_sigma_px=4"])
    assert cfg2.orientation.enabled and cfg2.uncertainty.model_sigma_px == 4.0
    data = json.loads(Path("configs/upgrade_recommended.json").read_text())
    for section, values in data.items():
        for key, value in values.items():
            assert VARIANTS["REC"][1][f"{section}.{key}"] == value
