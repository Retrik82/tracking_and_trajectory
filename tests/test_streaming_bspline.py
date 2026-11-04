import numpy as np
import pytest

from reconstruction.streaming_bspline import StreamingBSpline, StreamingSplineSettings

FRAMES = np.arange(0, 300, dtype=float)


def path(t):
    t = np.asarray(t, dtype=float)
    return np.c_[50 * t, 1500 * np.sin(t / 60), 300 + 0.005 * t ** 2]


def velocity(t):
    t = np.asarray(t, dtype=float)
    return np.c_[np.full_like(t, 50.0), 25 * np.cos(t / 60), 0.01 * t]


def noisy(seed=0, sigma=30.0):
    rng = np.random.default_rng(seed)
    return path(FRAMES) + rng.normal(0, sigma, (len(FRAMES), 3))


def run(settings, pts, frames=FRAMES, update_every=None):
    sp = StreamingBSpline(settings)
    latest = {}
    for f, p in zip(frames, pts):
        sp.add(f, p, weight=1.0)
        sp.update(f)
        smp = sp.latest(f)
        if smp is not None:
            latest[f] = smp
    sp.wait_idle()
    sp.close()
    return sp, latest


@pytest.mark.parametrize("solver", ["irls", "least_squares"])
def test_smoothing_reduces_noise_and_gives_velocity(solver):
    pts = noisy()
    s = StreamingSplineSettings(window_frames=45, update_every_frames=4, lag_frames=5, solver=solver)
    sp, latest = run(s, pts)
    keys = sorted(latest)
    frames = np.array([latest[f].frame for f in keys])          # точка отстаёт на lag_frames
    assert np.all(np.array(keys) - frames == s.lag_frames)
    est = np.array([latest[f].position for f in keys])
    err_s = np.linalg.norm(est - path(frames), axis=1)
    err_raw = np.linalg.norm(pts[frames.astype(int)] - path(frames), axis=1)
    assert np.median(err_s) < 0.5 * np.median(err_raw)
    vel = np.array([latest[f].velocity for f in keys])
    assert np.median(np.linalg.norm(vel - velocity(frames), axis=1)) < 8.0
    hist = sp.history()
    assert any(smp.final for smp in hist.values())
    st = sp.stats()
    assert st["fits"] > 10 and st["elapsed_ms_p95"] < 200


def test_causal_and_frozen_history():
    pts = noisy(seed=1)
    s = StreamingSplineSettings(window_frames=30, update_every_frames=3, lag_frames=4)
    full, latest_full = run(s, pts)
    cut = 150
    part, latest_part = run(s, pts[:cut], FRAMES[:cut])
    # оценка «latest» на тике f зависит только от измерений ≤ f
    for f in sorted(latest_part):
        np.testing.assert_allclose(latest_part[f].position, latest_full[f].position)
    # замороженные кадры у обрезанного прогона совпадают с полным
    hf, hp = full.history(), part.history()
    for f, smp in hp.items():
        if smp.final:
            np.testing.assert_allclose(smp.position, hf[f].position)
            assert hf[f].final


def test_gap_and_weights():
    pts = noisy(seed=2)
    s = StreamingSplineSettings(window_frames=45, update_every_frames=2, lag_frames=3, min_points=6)
    sp = StreamingBSpline(s)
    at_135 = in_gap = None
    for f, p in zip(FRAMES, pts):
        if 100 <= f < 130:
            continue                                     # провал измерений
        sp.add(f, p, covariance=np.eye(3) * (900.0 if f % 2 else 100.0))
        sp.update(f)
        if f == 135:
            at_135 = sp.latest(f)
            in_gap = sp.sample(115.0)                    # внутри окна сплайн определён и в провале
    assert at_135 is not None and np.linalg.norm(at_135.position - path(132)[0]) < 200
    assert in_gap is not None and np.linalg.norm(in_gap.position - path(115)[0]) < 400


def test_background_thread_delivers_results():
    pts = noisy(seed=3)
    s = StreamingSplineSettings(window_frames=45, update_every_frames=3, lag_frames=5, background=True)
    sp = StreamingBSpline(s)
    import time
    for f, p in zip(FRAMES, pts):
        sp.add(f, p)
        sp.update(f)
        time.sleep(0.002)                                # как в реальном потоке: кадры приходят не мгновенно
    sp.wait_idle()
    smp = sp.latest(FRAMES[-1])
    sp.close()
    assert smp is not None and np.linalg.norm(smp.position - path(FRAMES[-1] - 5)[0]) < 100
    assert sp.stats()["fits"] > 5


def test_deadline_adapts_nfev():
    s = StreamingSplineSettings(solver="least_squares", deadline_ms=0.001, max_nfev=25)
    sp = StreamingBSpline(s)
    for f, p in zip(FRAMES[:60], noisy()[:60]):
        sp.add(f, p)
        sp.update(f, force=True)
    assert sp._nfev < 25 and sp.stats()["deadline_violations"] > 0
