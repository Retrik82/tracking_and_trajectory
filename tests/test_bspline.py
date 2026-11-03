import numpy as np
import pytest

from reconstruction.bspline import BSplineTrajectory, fit_initial_bspline

T = np.arange(0, 1001, dtype=float)


def curve(t):
    return np.c_[1000 + 10 * t, 5000 + 3000 * np.sin(t / 100), 300 + 0.001 * t ** 2]


def test_fits_smooth_curve():
    spline = fit_initial_bspline(T, curve(T))
    assert np.abs(spline.evaluate(T) - curve(T)).max() < 1.0


def test_evaluate_shapes_and_bounds():
    spline = fit_initial_bspline(T, curve(T))
    assert spline.evaluate(5.0).shape == (3,)
    assert spline.evaluate(np.array([1.0, 2.0])).shape == (2, 3)
    np.testing.assert_allclose(spline.evaluate(T[-1]), curve(T[-1:])[0], atol=1.0)
    with pytest.raises(ValueError):
        spline.evaluate(T[-1] + 1)
    with pytest.raises(ValueError):
        spline.evaluate(np.nan)


def test_derivatives_match_analytic():
    spline = fit_initial_bspline(T, curve(T), knot_spacing=20)
    t = np.array([100.0, 400.0, 800.0])
    velocity = np.c_[np.full(3, 10.0), 30 * np.cos(t / 100), 0.002 * t]
    acceleration = np.c_[np.zeros(3), -0.3 * np.sin(t / 100), np.full(3, 0.002)]
    np.testing.assert_allclose(spline.derivative(1).evaluate(t), velocity, atol=0.05)
    np.testing.assert_allclose(spline.derivative(2).evaluate(t), acceleration, atol=0.01)


def test_gap_without_observations_is_bridged():
    # Раньше lstsq без регуляризации давал для пустых интервалов узлов
    # контрольные точки около нуля, и сплайн нырял в начало координат
    mask = (T < 300) | (T > 600)
    spline = fit_initial_bspline(T[mask], curve(T[mask]))
    inside = spline.evaluate(np.arange(300, 601, 10.0))
    truth = curve(np.arange(300, 601, 10.0))
    # Внутри разрыва сплайн не знает формы кривой, но должен оставаться в её окрестности
    assert np.abs(inside - truth).max() < 1000
    assert np.linalg.norm(spline.evaluate(450.0)) > 1000


def test_huber_resists_outliers():
    rng = np.random.default_rng(0)
    points = curve(T) + rng.normal(0, 5, (len(T), 3))
    points[[100, 500, 501, 900]] += [50000, -30000, 20000]
    plain = fit_initial_bspline(T, points)
    robust = fit_initial_bspline(T, points, loss="huber", f_scale=20)
    err_plain = np.abs(plain.evaluate(T) - curve(T)).max()
    err_robust = np.abs(robust.evaluate(T) - curve(T)).max()
    assert err_plain > 500
    assert err_robust < 20


def test_weights_zero_ignores_point():
    points = curve(T)
    points[500] += 10000
    weights = np.ones(len(T))
    weights[500] = 0
    spline = fit_initial_bspline(T, points, weights=weights)
    assert np.abs(spline.evaluate(T) - curve(T)).max() < 1.0


def test_knot_spacing_sets_control_points():
    spline = fit_initial_bspline(T, curve(T), knot_spacing=50)
    assert len(spline.control_points) == 1000 // 50 + 3
    with pytest.raises(ValueError):
        fit_initial_bspline(T, curve(T), knot_spacing=50, num_control_points=10)


@pytest.mark.parametrize("bad", [
    dict(times=T[::-1]),
    dict(points_3d=np.full((len(T), 3), np.nan)),
    dict(weights=-np.ones(len(T))),
    dict(loss="cauchy"),
])
def test_invalid_input(bad):
    kwargs = dict(times=T, points_3d=curve(T))
    kwargs.update(bad)
    with pytest.raises(ValueError):
        fit_initial_bspline(**kwargs)


def test_trajectory_validates_knots():
    with pytest.raises(ValueError):
        BSplineTrajectory(knots=np.zeros(5), control_points=np.zeros((4, 3)), degree=3, t_start=0, t_end=1)
