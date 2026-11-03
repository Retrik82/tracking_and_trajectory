"""
«Обычная аппроксимация» 3D-точек: базовый вариант для сравнения с трекером.

Вход — точки триангуляции центров масок по кадрам (часть кадров без точки: NaN).
Никакой модели грузовика, никаких рамок, никакого фильтра — только подгонка полинома
к самим 3D-точкам. Два варианта:

- causal_polynomial — РЕАЛЬНОЕ ВРЕМЯ: на кадре t полином степени degree подгоняется
  методом наименьших квадратов к точкам окна (t − window, t] (только прошлое) и
  вычисляется в t; скорость — производная полинома. Если в окне мало точек, положение
  экстраполируется по прямой: последнее положение полинома и скорость из линейной подгонки
  по тому же окну (производная параболы по 5–8 точкам слишком шумная, а квадратичный член
  за десятки кадров уводит точку на сотни метров), не дольше max_extrapolation кадров.
  Робастность: итеративно перевзвешенные наименьшие квадраты с весом Хьюбера.
- global_polynomial — ОФЛАЙН: один полином степени degree по всем точкам видеоряда
  (заглядывает вперёд; показывает, почему глобальная аппроксимация не годится для петли).

Так видно, что даёт сама идея «сгладить 3D-точки» без модели объекта, и в чём трекер
её превосходит (систематическое смещение центра маски, скачки при смене камер, провалы
без наблюдений).
"""

from dataclasses import dataclass

import numpy as np

PREDICTED, MEASURED, NONE = "predicted", "measured", "none"


@dataclass
class ApproximationResult:
    frames: np.ndarray            # (N,)
    points: np.ndarray            # (N, 3) положение, NaN где оценки нет
    velocity: np.ndarray          # (N, 3) ед. мира / кадр
    status: np.ndarray            # (N,) measured | predicted | none
    points_used: np.ndarray       # (N,) сколько точек вошло в подгонку на кадре


def _vandermonde(dt: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(dt, degree + 1, increasing=True)


def _fit_poly(dt: np.ndarray, pts: np.ndarray, degree: int, huber_scale: float, iterations: int) -> np.ndarray:
    """Коэффициенты (degree + 1, 3) полинома по dt (0 — текущий кадр), IRLS с весом Хьюбера."""
    A = _vandermonde(dt, degree)
    w = np.ones(len(dt))
    coef = None
    for _ in range(max(1, iterations)):
        sw = np.sqrt(w)[:, None]
        new = np.linalg.lstsq(A * sw, pts * sw, rcond=None)[0]
        if coef is not None and np.allclose(new, coef, atol=1e-6):
            coef = new
            break
        coef = new
        r = np.linalg.norm(A @ coef - pts, axis=1)
        w = np.minimum(1.0, huber_scale / np.maximum(r, 1e-12))
    return coef


def causal_polynomial(frames: np.ndarray, points: np.ndarray, window: int = 15, degree: int = 2,
                      min_points: int = 8, max_extrapolation: int = 30, huber_scale: float = 100.0,
                      iterations: int = 5) -> ApproximationResult:
    """
    Аппроксимация в реальном времени: полином по окну прошлых точек.

    frames — (N,) кадры по порядку; points — (N, 3), NaN где точки нет.
    window — длина окна в кадрах; degree — степень полинома (1 — прямая, 2 — парабола);
    min_points — минимум точек в окне для подгонки; max_extrapolation — сколько кадров
    держать прогноз последним полиномом, когда точек не хватает; huber_scale — порог
    Хьюбера в единицах мира.
    """
    frames = np.asarray(frames, dtype=float)
    points = np.asarray(points, dtype=float)
    n = len(frames)
    if points.shape != (n, 3):
        raise ValueError("points должен иметь форму (N, 3)")
    if window < 1 or degree < 0 or min_points < degree + 1:
        raise ValueError("window >= 1, degree >= 0, min_points >= degree + 1")
    ok = np.all(np.isfinite(points), axis=1)
    out_p = np.full((n, 3), np.nan)
    out_v = np.full((n, 3), np.nan)
    status = np.full(n, NONE, dtype=object)
    used = np.zeros(n, dtype=int)
    last = None                                    # (кадр, положение, скорость) последней удачной подгонки
    for i in range(n):
        t = frames[i]
        lo = np.searchsorted(frames, t - window, side="right")
        idx = np.nonzero(ok[lo:i + 1])[0] + lo
        if len(idx) >= min_points:
            coef = _fit_poly(frames[idx] - t, points[idx], degree, huber_scale, iterations)
            out_p[i] = coef[0]
            out_v[i] = coef[1] if degree >= 1 else 0.0
            status[i] = MEASURED
            used[i] = len(idx)
            linear = coef if degree <= 1 else _fit_poly(frames[idx] - t, points[idx], 1, huber_scale, iterations)
            last = (t, coef[0], linear[1])
        elif last is not None and t - last[0] <= max_extrapolation:
            dt = t - last[0]
            out_p[i] = last[1] + dt * last[2]
            out_v[i] = last[2]
            status[i] = PREDICTED
    return ApproximationResult(frames, out_p, out_v, status.astype(str), used)


def global_polynomial(frames: np.ndarray, points: np.ndarray, degree: int = 8) -> ApproximationResult:
    """Офлайн: один полином степени degree по всем точкам (время нормировано на [-1, 1])."""
    frames = np.asarray(frames, dtype=float)
    points = np.asarray(points, dtype=float)
    ok = np.all(np.isfinite(points), axis=1)
    if ok.sum() < degree + 1:
        raise ValueError("Слишком мало точек для полинома такой степени")
    t0, t1 = frames[ok].min(), frames[ok].max()
    scale = (t1 - t0) / 2 or 1.0
    s = (frames - (t0 + t1) / 2) / scale
    coef = np.linalg.lstsq(_vandermonde(s[ok], degree), points[ok], rcond=None)[0]
    inside = (frames >= t0) & (frames <= t1)
    pts = np.full((len(frames), 3), np.nan)
    vel = np.full((len(frames), 3), np.nan)
    pts[inside] = _vandermonde(s[inside], degree) @ coef
    dcoef = coef[1:] * np.arange(1, degree + 1)[:, None] / scale
    vel[inside] = _vandermonde(s[inside], degree - 1) @ dcoef if degree >= 1 else 0.0
    status = np.where(inside, MEASURED, NONE)
    return ApproximationResult(frames, pts, vel, status.astype(str), np.where(inside, int(ok.sum()), 0))
