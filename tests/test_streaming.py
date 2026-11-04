"""Потоковый конвейер: причинность, синхронизация, воспроизведение видео, отчёт о производительности."""

import csv
import math
import time
from pathlib import Path

import numpy as np
import pytest

from detection.schema import Detection
from pipeline.config import StreamConfig
from pipeline.streaming import (
    DetectionSource, Frame, FrameSource, PrecomputedDetector, StreamPipeline, Synchronizer, VideoSource,
    load_detections_table,
)
from reconstruction.cab_model import DEFAULT_DIMENSIONS, box_corners
from reconstruction.camera import Camera
from tests.test_visibility import CAMERAS as TEST_CAMERAS

NUM = 120
FPS = 30.0


def path(t):
    t = np.asarray(t, dtype=float)
    return np.c_[50 * t, 1500 * np.sin(t / 120), 300 + 0 * t]


def velocity(t):
    t = np.asarray(t, dtype=float)
    return np.c_[np.full_like(t, 50.0), 1500 / 120 * np.cos(t / 120), 0 * t]


def make_table(seed=0, drop=None) -> dict[int, dict[int, Detection]]:
    rng = np.random.default_rng(seed)
    table = {i: {} for i in range(len(TEST_CAMERAS))}
    for f in range(NUM):
        corners = box_corners(path([f]), velocity([f]), DEFAULT_DIMENSIONS.as_array())[0]
        for i, cam in enumerate(TEST_CAMERAS):
            if drop is not None and drop(i, f):
                continue
            uv = cam.project(corners)
            x1, x2, y1, y2 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
            c = cam.project(corners.mean(axis=0))
            table[i][f] = Detection(i, f, 0.9, float(x1 + rng.normal(0, 0.5)), float(y1), float(x2), float(y2),
                                    float(c[0]), float(c[1]), "mask_centroid", 5000.0, float(c[0]), float(c[1]))
    return table


def pose_provider(cam, index):
    return TEST_CAMERAS[cam]


def base_config(**over) -> StreamConfig:
    cfg = StreamConfig()
    cfg.cameras.ids = list(range(len(TEST_CAMERAS)))
    cfg.streaming.mode = "detections"
    cfg.streaming.realtime_pacing = False
    cfg.streaming.log_every = 0
    cfg.spline.background = False
    cfg.tracker.edge_sigma_px = 2.0
    cfg.tracker.truck_fbx = "models/NO_SUCH_MODEL.fbx"   # синтетика — коробка: реальный FBX из models/ не подхватывать
    cfg.output.performance_report = True
    for k, v in over.items():
        section, key = k.split(".")
        setattr(getattr(cfg, section), key, v)
    return cfg


def run_pipeline(cfg, table, out, num=NUM, **kw):
    pipe = StreamPipeline(cfg, Path("."), out, table, num, model=None, pose_provider=pose_provider, **kw)
    pipe.run()
    return pipe


def test_detections_mode_tracks_and_reports(tmp_path):
    pipe = run_pipeline(base_config(), make_table(), tmp_path)
    recs = pipe.records
    assert len(recs) == NUM
    pos = np.array([r.reference for r in recs])
    err = np.linalg.norm(pos - path(np.arange(NUM)), axis=1)
    assert np.all(np.isfinite(pos[30:])) and np.nanmedian(err[40:]) < 60
    assert (tmp_path / "stream_track.csv").exists() and (tmp_path / "performance_report.md").exists()
    perf = pipe.performance()
    assert perf["ticks"] == NUM and perf["latency_ms"]["p95"] > 0 and "tracker" in perf["stages"]
    assert any(np.isfinite(r.smoothed_index) for r in recs[60:])
    with open(tmp_path / "stream_track.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == NUM and rows[-1]["status"] in ("measured", "partial", "predicted")


def test_causal_truncated_run_matches_prefix(tmp_path):
    table = make_table(seed=1)
    full = run_pipeline(base_config(), table, tmp_path / "full")
    cut = 70
    short_table = {c: {f: d for f, d in t.items() if f < cut} for c, t in table.items()}
    short = run_pipeline(base_config(), short_table, tmp_path / "short", num=cut)
    for a, b in zip(short.records, full.records[:cut]):
        np.testing.assert_array_equal(a.reference, b.reference)
        np.testing.assert_array_equal(a.smoothed_position, b.smoothed_position)


def test_scheduler_reduces_detections_without_losing_track(tmp_path):
    table = make_table(seed=2)
    on = run_pipeline(base_config(**{"scheduler.enabled": True, "scheduler.min_active": 1,
                                     "scheduler.probe_period_frames": 10}), table, tmp_path / "on")
    off = run_pipeline(base_config(**{"scheduler.enabled": False}), table, tmp_path / "off")
    assert on.detector.images <= off.detector.images
    e_on = np.linalg.norm(np.array([r.reference for r in on.records[40:]]) - path(np.arange(40, NUM)), axis=1)
    e_off = np.linalg.norm(np.array([r.reference for r in off.records[40:]]) - path(np.arange(40, NUM)), axis=1)
    assert np.nanmedian(e_on) < 3 * max(np.nanmedian(e_off), 20.0)


def test_loss_and_reappearance_statuses(tmp_path):
    table = make_table(seed=3, drop=lambda cam, f: 60 <= f <= 75)
    pipe = run_pipeline(base_config(), table, tmp_path)
    st = [r.status for r in pipe.records]
    assert all(s == "predicted" for s in st[60:76]) and st[80] == "measured"
    assert all(np.all(np.isfinite(r.reference)) for r in pipe.records[60:76])


def test_synchronizer_excludes_desynchronized_camera(tmp_path):
    table = make_table(seed=4)
    pipe = run_pipeline(base_config(**{"streaming.sync_tolerance_ms": 20.0}), table, tmp_path,
                        source_delays={1: 2})           # метки камеры 1 на 2 кадра (67 мс) впереди остальных
    excluded = [r.index for r in pipe.records if 1 in r.excluded]
    # кадр «из будущего» исключается из связки, пока остальные камеры не дойдут до его метки (2 тика),
    # дальше камера стыкуется по меткам и больше не исключается; её кадры при этом не теряются
    assert excluded and max(excluded) <= 2
    assert all(1 not in r.detected_cameras for r in pipe.records if 1 in r.excluded)
    assert sum(1 in r.detected_cameras for r in pipe.records) >= NUM - 3
    ok = run_pipeline(base_config(**{"streaming.sync_tolerance_ms": 80.0}), table, tmp_path / "tol",
                      source_delays={1: 2})
    assert sum(1 in r.excluded for r in ok.records) == 0


def test_synchronizer_keeps_future_frame_for_next_tick():
    """Камера, ушедшая на кадр вперёд (разный старт источников), на следующем тике стыкуется, а не
    теряет отложенный кадр и не остаётся впереди навсегда (ошибка, найденная на реальном replay)."""
    a, b = FrameSource(1, queue_size=4, lossless=True), FrameSource(2, queue_size=4, lossless=True)
    img = np.zeros((2, 2, 3), np.uint8)
    for i in (1, 2, 3):                          # камера 1 стартовала раньше: кадр 0 уже ушёл в одиночную связку
        a.push(Frame(1, i, i / FPS, img, time.perf_counter()))
    for i in (0, 1, 2, 3):
        b.push(Frame(2, i, i / FPS, img, time.perf_counter()))
    sync = Synchronizer({1: a, 2: b}, FPS, tolerance_s=0.02, max_age_s=0.15, wait_s=0.05)
    first = sync.next_bundle()
    assert first.index == 0 and set(first.frames) == {2} and 1 in first.excluded
    for expected in (1, 2, 3):
        bundle = sync.next_bundle()
        assert bundle.index == expected and set(bundle.frames) == {1, 2} and not bundle.excluded
        assert {f.index for f in bundle.frames.values()} == {expected}
    a.done = b.done = True
    assert sync.next_bundle() is None


def test_frame_source_drops_old_frames_when_not_lossless():
    src = FrameSource(0, queue_size=2)
    for i in range(5):
        src.push(Frame(0, i, i / FPS, np.zeros((2, 2, 3), np.uint8), time.perf_counter()))
    assert src.dropped == 3 and src.pop_latest().index == 4
    stale = Frame(0, 9, 0.3, np.zeros((2, 2, 3), np.uint8), time.perf_counter() - 1.0)
    src.push(stale)
    assert src.pop_latest(max_age_s=0.1) is None and src.dropped == 4


@pytest.mark.skipif(pytest.importorskip("cv2") is None, reason="нет OpenCV")
def test_video_replay_with_pacing_and_dropping(tmp_path):
    import cv2

    n, fps, w, h = 24, 30.0, 320, 180
    paths = []
    for cam in range(2):
        p = tmp_path / f"cam{cam}.mp4"
        writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(n):
            img = np.zeros((h, w, 3), np.uint8)
            img[:, : (i * 10) % w] = (0, 120, 255)
            writer.write(img)
        writer.release()
        paths.append(p)
    t0 = time.perf_counter() + 0.05
    sources = {cam: VideoSource(cam, str(p), fps, t0, pacing=True, queue_size=2) for cam, p in enumerate(paths)}
    for s in sources.values():
        s.start()
    sync = Synchronizer(sources, fps, 0.02, 0.5)
    bundles = []
    started = time.perf_counter()
    while True:
        b = sync.next_bundle()
        if b is None:
            if sync.all_done():
                break
            continue
        bundles.append(b)
        if len(bundles) == 5:
            time.sleep(0.25)                                # потребитель «завис»: старые кадры должны вытесниться
    elapsed = time.perf_counter() - started
    for s in sources.values():
        s.stop()
    idx = [b.index for b in bundles]
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    # обычно обе камеры в связке; после «зависания» допустимы одиночные связки, пока камеры не выровняются
    assert sum(len(b.frames) == 2 for b in bundles) >= 0.6 * len(bundles)
    assert sum(s.dropped for s in sources.values()) > 0    # вытеснение при зависшем потребителе
    assert elapsed >= (n - 1) / fps * 0.8                   # воспроизведение в исходном темпе, не быстрее


def test_detections_mode_runs_with_pacing(tmp_path):
    cfg = base_config(**{"streaming.realtime_pacing": True})
    started = time.perf_counter()
    pipe = run_pipeline(cfg, make_table(seed=5), tmp_path, num=30)
    assert time.perf_counter() - started >= 29 / FPS * 0.8
    assert len(pipe.records) >= 25


def test_load_detections_table(tmp_path):
    table = make_table(seed=6)
    for cam, rows in table.items():
        with open(tmp_path / f"cam{cam}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=Detection.columns())
            w.writeheader()
            w.writerows(d.to_row() for d in rows.values())
    loaded, n = load_detections_table(tmp_path, list(table))
    assert n == NUM and loaded[0][5].x1 == table[0][5].x1
