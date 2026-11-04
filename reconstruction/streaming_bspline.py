"""
Скользящий B-сплайн в реальном времени (ТЗ 3.4): SMOOTHED TRACK с фиксированной задержкой.

На каждом обновлении (раз в update_every_frames) по измерениям окна [now − window, now]
(только прошлое) подгоняется кубический B-сплайн с узлами через knot_spacing кадров:

    min Σ_i w_i ρ_H(|B(t_i) c − p_i|) + λ_a ∫|P''|² dt + λ_j ∫|P'''|² dt

w_i — вес измерения (из ковариации фильтра: 1/σ_i², нормированный), ρ_H — функция Хьюбера.
Та же постановка, что в офлайн-оптимизаторе (regularization.py), но при фиксированных весах
задача ЛИНЕЙНА по контрольным точкам: остатки = A c − b с постоянной матрицей A (базис, его
вторая и третья производные). Поэтому вместо численного least_squares с конечными разностями
она решается точно — нормальными уравнениями с итеративным перевзвешиванием Хьюбера (IRLS):
за доли миллисекунды и с гарантированным бюджетом времени. Для сравнения оставлен solver =
"least_squares" (scipy, с аналитическим якобианом, max_nfev и адаптацией под deadline).

Причинность: сплайн на тике now видит только измерения ≤ now. Выход:
  latest(now)  — точка в момент now − lag_frames (внутри окна, уже «стабилизированная»);
  history()    — замороженные оценки для кадров, вышедших из окна (пересчёт прекращается),
                 плюс текущее окно. Производные сплайна дают сглаженные скорость и ускорение.
Фоновый режим: вычисления в отдельном потоке, очередь из одного запроса (новый заменяет
старый), основной контур не блокируется; результат читается под блокировкой.
"""

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline

from reconstruction.bspline import BSplineTrajectory, _create_open_uniform_knots

DEGREE = 3


@dataclass
class StreamingSplineSettings:
    window_frames: int = 45
    update_every_frames: int = 8
    lag_frames: int = 5
    knot_spacing_frames: float = 6.0
    lambda_acceleration: float = 1e3       # (см, кадры): подобрано на синтетике из 1e3 / 1e4 / 1e5 по ошибке скорости и ускорения
    lambda_jerk: float = 1e5
    huber_scale: float = 100.0             # ед. мира
    max_nfev: int = 25                     # для solver = least_squares
    deadline_ms: float = 25.0
    background: bool = False
    solver: str = "irls"                   # irls | least_squares
    irls_iterations: int = 6
    min_points: int = 6
    regularization_step: float = 1.0       # кадров между точками квадратуры регуляризации


@dataclass
class SmoothedSample:
    frame: float
    position: np.ndarray                   # ед. мира
    velocity: np.ndarray                   # ед. мира / кадр
    acceleration: np.ndarray               # ед. мира / кадр²
    final: bool = False                    # кадр вышел из окна, оценка больше не меняется
    fitted_at: float = float("nan")        # тик, на котором получена оценка (задержка = fitted_at − frame)


@dataclass
class FitInfo:
    frame: float
    points: int
    elapsed_ms: float
    iterations: int
    converged: bool
    residual_rms: float


def _design(times: np.ndarray, knots: np.ndarray, k: int, nu: int = 0) -> np.ndarray:
    """Матрица базиса (len(times), K) или его производной порядка nu."""
    K = len(knots) - k - 1
    if nu == 0:
        return BSpline.design_matrix(times, knots, k, extrapolate=False).toarray()
    return BSpline(knots, np.eye(K), k, extrapolate=False).derivative(nu)(times)


class StreamingBSpline:
    def __init__(self, settings: StreamingSplineSettings | None = None):
        self.s = settings or StreamingSplineSettings()
        self._frames: list[float] = []
        self._points: list[np.ndarray] = []
        self._weights: list[float] = []
        self._lock = threading.Lock()
        self._spline: BSplineTrajectory | None = None
        self._spline_fitted_at = float("nan")
        self._frozen: dict[float, SmoothedSample] = {}
        self._last_update: float = -np.inf
        self._last_requested: float = -np.inf
        self.fits: list[FitInfo] = []
        self._nfev = self.s.max_nfev
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        if self.s.background:
            self._queue = queue.Queue(maxsize=1)
            self._thread = threading.Thread(target=self._worker, name="streaming-bspline", daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------- данные

    def add(self, frame: float, point: np.ndarray, covariance: np.ndarray | None = None,
            weight: float | None = None) -> None:
        """Измерение на кадре frame; вес из ковариации (1/средняя дисперсия) или явно."""
        point = np.asarray(point, dtype=float)
        if not np.all(np.isfinite(point)):
            return
        if weight is None:
            if covariance is not None and np.all(np.isfinite(covariance)):
                var = max(float(np.trace(covariance)) / 3, 1e-6)
                weight = 1.0 / var
            else:
                weight = 1.0
        with self._lock:
            if self._frames and frame <= self._frames[-1]:
                return                                           # измерения только вперёд по времени
            self._frames.append(float(frame))
            self._points.append(point)
            self._weights.append(float(weight))
            self._trim(frame)

    def _trim(self, now: float) -> None:
        """Кадры, вышедшие из окна, замораживаются по последнему сплайну и удаляются из буфера."""
        lo = now - self.s.window_frames
        while self._frames and self._frames[0] < lo:
            f = self._frames.pop(0)
            self._points.pop(0)
            self._weights.pop(0)
            if f not in self._frozen:
                sample = self._evaluate_locked(f)
                if sample is not None:
                    sample.final = True
                    self._frozen[f] = sample
            else:
                self._frozen[f].final = True

    # ---------------------------------------------------------------- подгонка

    def _fit_locked(self, now: float) -> FitInfo | None:
        s = self.s
        frames = np.array(self._frames)
        if len(frames) < s.min_points or frames[-1] - frames[0] < 2:
            return None
        pts = np.array(self._points)
        w = np.array(self._weights)
        w = w / w.mean()
        t0, t1 = frames[0], frames[-1]
        n_intervals = max(1, int(np.ceil((t1 - t0) / s.knot_spacing_frames)))
        K = n_intervals + DEGREE
        knots = _create_open_uniform_knots(K, DEGREE, t0, t1)
        B = _design(frames, knots, DEGREE)
        tq = np.arange(t0, t1 + 1e-9, s.regularization_step)
        if len(tq) < 2:
            tq = np.array([t0, t1])
        hq = np.gradient(tq) if len(tq) > 1 else np.ones(1)
        A_acc = np.sqrt(s.lambda_acceleration * hq)[:, None] * _design(tq, knots, DEGREE, 2)
        A_jerk = np.sqrt(s.lambda_jerk * hq)[:, None] * _design(tq, knots, DEGREE, 3)
        A_reg = np.vstack([A_acc, A_jerk])
        started = time.perf_counter()
        if s.solver == "least_squares":
            c, iters, converged = self._solve_least_squares(B, pts, w, A_reg)
        else:
            c, iters, converged = self._solve_irls(B, pts, w, A_reg)
        elapsed = (time.perf_counter() - started) * 1000
        self._spline = BSplineTrajectory(knots, c, DEGREE, float(t0), float(t1))
        self._spline_fitted_at = now
        self._last_update = now
        rms = float(np.sqrt(np.mean(np.sum((B @ c - pts) ** 2, axis=1))))
        info = FitInfo(now, len(frames), elapsed, iters, converged, rms)
        self.fits.append(info)
        # адаптация бюджета для least_squares
        if elapsed > s.deadline_ms:
            self._nfev = max(3, self._nfev // 2)
        elif elapsed < s.deadline_ms / 3:
            self._nfev = min(s.max_nfev, self._nfev + 2)
        return info

    def _solve_irls(self, B, pts, w, A_reg):
        s = self.s
        K = B.shape[1]
        robust = np.ones(len(w))
        c = None
        reg = A_reg.T @ A_reg
        for it in range(max(1, s.irls_iterations)):
            ww = w * robust
            normal = B.T @ (ww[:, None] * B) + reg + np.eye(K) * 1e-9
            rhs = B.T @ (ww[:, None] * pts)
            new = np.linalg.solve(normal, rhs)
            converged = c is not None and np.allclose(new, c, atol=1e-3)
            c = new
            if converged:
                return c, it + 1, True
            r = np.linalg.norm(B @ c - pts, axis=1)
            robust = np.minimum(1.0, s.huber_scale / np.maximum(r, 1e-12))
        return c, s.irls_iterations, False

    def _solve_least_squares(self, B, pts, w, A_reg):
        from scipy.optimize import least_squares

        s = self.s
        K = B.shape[1]
        sw = np.sqrt(w)[:, None]
        x0 = (self._warm_start(B, pts, w, A_reg, K)).ravel()
        A_data = np.kron(np.eye(3), sw * B)                       # остатки по осям: 3N
        A_full = np.vstack([A_data, np.kron(np.eye(3), A_reg)])
        b = np.concatenate([(sw * pts).T.ravel(), np.zeros(3 * A_reg.shape[0])])

        def residuals(x):
            return A_full @ x - b

        res = least_squares(residuals, x0, jac=lambda x: A_full, loss="huber", f_scale=s.huber_scale,
                            max_nfev=max(3, self._nfev), method="trf")
        return res.x.reshape(3, K).T, int(res.nfev), bool(res.success)

    def _warm_start(self, B, pts, w, A_reg, K):
        """Начальное приближение: предыдущий сплайн (на новой сетке) или взвешенное решение без Хьюбера."""
        normal = B.T @ (w[:, None] * B) + A_reg.T @ A_reg + np.eye(K) * 1e-9
        c0 = np.linalg.solve(normal, B.T @ (w[:, None] * pts))
        if self._spline is None:
            return c0.T
        # значения прежнего сплайна там, где он определён, подменяют данные: решение ближе к старому
        frames = np.array(self._frames)
        inside = (frames >= self._spline.t_start) & (frames <= self._spline.t_end)
        if inside.sum() < K:
            return c0.T
        prev = pts.copy()
        prev[inside] = self._spline.evaluate(frames[inside])
        c1 = np.linalg.solve(normal, B.T @ (w[:, None] * prev))
        return c1.T

    # ---------------------------------------------------------------- обновление

    def update(self, now: float, force: bool = False) -> bool:
        """Запрос обновления на тике now. Фоновый режим: ставит в очередь; иначе считает сразу."""
        if not force and now - self._last_requested < self.s.update_every_frames:
            return False
        self._last_requested = now
        if self._queue is not None:
            try:
                self._queue.put_nowait(now)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(now)
            return True
        with self._lock:
            return self._fit_locked(now) is not None

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                now = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            with self._lock:
                try:
                    self._fit_locked(now)
                except Exception:  # noqa: BLE001 — фоновый поток не должен падать
                    pass

    def wait_idle(self, timeout: float = 2.0) -> None:
        """Дождаться, пока очередь фонового потока опустеет (тесты, завершение)."""
        if self._queue is None:
            return
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.002)
        with self._lock:
            pass

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ---------------------------------------------------------------- выход

    def _evaluate_locked(self, frame: float, extrapolate: float = 0.0) -> SmoothedSample | None:
        """
        Точка сплайна на кадре frame. Между обновлениями конец сплайна отстаёт от now на
        ≤ update_every кадров; запрос за концом (не дальше extrapolate кадров) продолжается по
        прямой со скоростью в конце сплайна — это прогноз, не сглаженное измерение.
        """
        sp = self._spline
        if sp is None or frame < sp.t_start or frame > sp.t_end + extrapolate:
            return None
        if frame <= sp.t_end:
            t = np.array([frame])
            return SmoothedSample(frame, sp.evaluate(t)[0], sp.derivative(1).evaluate(t)[0],
                                  sp.derivative(2).evaluate(t)[0], False, self._spline_fitted_at)
        t = np.array([sp.t_end])
        p, v, a = sp.evaluate(t)[0], sp.derivative(1).evaluate(t)[0], sp.derivative(2).evaluate(t)[0]
        return SmoothedSample(frame, p + v * (frame - sp.t_end), v, a, False, self._spline_fitted_at)

    def sample(self, frame: float) -> SmoothedSample | None:
        with self._lock:
            if frame in self._frozen:
                return self._frozen[frame]
            return self._evaluate_locked(frame)

    def latest(self, now: float) -> SmoothedSample | None:
        """Сглаженная точка с задержкой lag_frames относительно now (за концом сплайна — по прямой)."""
        frame = now - self.s.lag_frames
        with self._lock:
            if frame in self._frozen:
                return self._frozen[frame]
            return self._evaluate_locked(frame, extrapolate=self.s.update_every_frames)

    def history(self) -> dict[float, SmoothedSample]:
        """Замороженные кадры + текущее окно (по последнему сплайну)."""
        with self._lock:
            out = dict(self._frozen)
            for f in self._frames:
                if f not in out:
                    smp = self._evaluate_locked(f)
                    if smp is not None:
                        out[f] = smp
            return dict(sorted(out.items()))

    def stats(self) -> dict:
        if not self.fits:
            return {"fits": 0}
        el = np.array([f.elapsed_ms for f in self.fits])
        return {"fits": len(self.fits), "elapsed_ms_mean": float(el.mean()), "elapsed_ms_p95": float(np.percentile(el, 95)),
                "elapsed_ms_max": float(el.max()), "points_mean": float(np.mean([f.points for f in self.fits])),
                "deadline_violations": int((el > self.s.deadline_ms).sum()),
                "solver": self.s.solver}
