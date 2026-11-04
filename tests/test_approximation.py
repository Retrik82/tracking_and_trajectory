import numpy as np
import pytest

from reconstruction.approximation import MEASURED, NONE, PREDICTED, causal_polynomial, global_polynomial

FRAMES = np.arange(0, 200, dtype=float)


def path(t):
    t = np.asarray(t, dtype=float)
    return np.c_[50 * t, 1500 * np.sin(t / 120), 300 + 0.01 * t ** 2]


def velocity(t):
    t = np.asarray(t, dtype=float)
    return np.c_[np.full_like(t, 50.0), 1500 / 120 * np.cos(t / 120), 0.02 * t]


def test_causal_polynomial_tracks_and_uses_only_past():
    rng = np.random.default_rng(0)
    pts = path(FRAMES) + rng.normal(0, 5, (len(FRAMES), 3))
    res = causal_polynomial(FRAMES, pts, window=15, degree=2)
    assert np.all(res.status[:4] == NONE) and np.all(res.status[20:] == MEASURED)
    err = np.linalg.norm(res.points - path(FRAMES), axis=1)
    assert np.nanmedian(err[20:]) < 10
    verr = np.linalg.norm(res.velocity - velocity(FRAMES), axis=1)
    assert np.nanmedian(verr[20:]) < 3
    # причинность: обрезанный ряд даёт те же оценки на общих кадрах
    cut = 100
    short = causal_polynomial(FRAMES[:cut], pts[:cut], window=15, degree=2)
    np.testing.assert_array_equal(short.points, res.points[:cut])


def test_causal_polynomial_gap_is_predicted_then_none():
    pts = path(FRAMES)
    pts[80:140] = np.nan
    res = causal_polynomial(FRAMES, pts, window=15, degree=2, min_points=5, max_extrapolation=30)
    # до кадра 89 в окне ещё ≥5 точек до провала; затем 30 кадров прогноза по прямой; потом ничего,
    # пока не наберётся 5 точек
    assert np.all(res.status[80:90] == MEASURED)
    assert np.all(res.status[90:120] == PREDICTED)
    assert np.all(res.status[120:144] == NONE)
    assert np.all(np.isfinite(res.points[90:120])) and np.all(np.isnan(res.points[120:144]))
    assert res.status[144] == MEASURED
    # прогноз по прямой: ошибка растёт не быстрее, чем кривизна пути за 30 кадров
    assert np.linalg.norm(res.points[119] - path(FRAMES[119])) < 100


def test_causal_polynomial_huber_resists_outlier():
    pts = path(FRAMES)
    pts[150] += [0, 3000, 0]
    plain = causal_polynomial(FRAMES, pts, window=15, degree=2, huber_scale=1e9)
    robust = causal_polynomial(FRAMES, pts, window=15, degree=2, huber_scale=50.0)
    err_plain = np.linalg.norm(plain.points[151:165] - path(FRAMES[151:165]), axis=1).max()
    err_robust = np.linalg.norm(robust.points[151:165] - path(FRAMES[151:165]), axis=1).max()
    assert err_robust < err_plain / 5


def test_global_polynomial_fits_smooth_path():
    pts = path(FRAMES)
    pts[7::7] = np.nan
    res = global_polynomial(FRAMES, pts, degree=6)
    err = np.linalg.norm(res.points - path(FRAMES), axis=1)
    assert np.nanmax(err) < 5 and np.all(res.status == MEASURED)
    with pytest.raises(ValueError):
        global_polynomial(FRAMES[:3], pts[:3], degree=5)


def test_bad_arguments():
    with pytest.raises(ValueError):
        causal_polynomial(FRAMES, np.zeros((10, 3)))
    with pytest.raises(ValueError):
        causal_polynomial(FRAMES, path(FRAMES), degree=2, min_points=2)
