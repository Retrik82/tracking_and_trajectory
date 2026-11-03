from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline


@dataclass
class BSplineTrajectory:
    """Непрерывная 3D-траектория, заданная B-сплайном в реальном времени."""

    knots: np.ndarray
    control_points: np.ndarray
    degree: int
    t_start: float
    t_end: float

    def __post_init__(self):
        self.knots = np.asarray(self.knots, dtype=float)
        self.control_points = np.asarray(self.control_points, dtype=float)

        if self.control_points.ndim != 2 or self.control_points.shape[1] != 3:
            raise ValueError("control_points должны иметь размерность (N, 3)")

        if self.degree < 0:
            raise ValueError("Степень сплайна не может быть отрицательной")

        expected_knots = len(self.control_points) + self.degree + 1
        if len(self.knots) != expected_knots:
            raise ValueError("Некорректная длина узлового вектора")

        if self.t_end <= self.t_start:
            raise ValueError("t_end должен быть больше t_start")

        # Инициализируем сплайн сразу на реальной временной сетке.
        # extrapolate=False вернет NaN для выхода за границы, 
        # но мы это валидируем сами.
        self._spline = BSpline(
            self.knots,
            self.control_points,
            self.degree,
            extrapolate=False,
        )

    def evaluate(self, time: float | np.ndarray) -> np.ndarray:
        """Вычисляет положение объекта в заданный момент времени."""
        time_arr = np.asarray(time, dtype=float)

        if not np.all(np.isfinite(time_arr)):
            raise ValueError("Время содержит NaN или inf")

        # Небольшой epsilon (1e-14) защищает от ошибок float на правых границах
        if np.any(time_arr < self.t_start) or np.any(time_arr > self.t_end + 1e-14):
            raise ValueError("Время находится за пределами траектории")

        # Ограничиваем сверху t_end, чтобы избежать NaN из-за погрешности float
        time_arr = np.clip(time_arr, self.t_start, self.t_end)

        # Переводить в [0, 1] больше не нужно, сплайн работает в физическом времени
        res = self._spline(time_arr)
        
        # Если передали скаляр float, возвращаем одномерный массив (3,) вместо (1, 3)
        if np.isscalar(time):
            return res.item() if res.size == 1 else res.squeeze()
        return res

    def derivative(self, nu: int = 1) -> "BSplineTrajectory":
        """Производная порядка nu как отдельная траектория (единицы: мир / время**nu)."""
        if not 1 <= nu <= self.degree:
            raise ValueError("Порядок производной должен быть от 1 до степени сплайна")
        spline = self._spline.derivative(nu)
        # scipy дополняет коэффициенты нулями до длины узлового вектора
        num_coefficients = len(spline.t) - spline.k - 1
        return BSplineTrajectory(
            knots=spline.t,
            control_points=spline.c[:num_coefficients],
            degree=spline.k,
            t_start=self.t_start,
            t_end=self.t_end,
        )

    def sample(self, num_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Вычисляет набор равномерно расположенных точек траектории."""
        if num_samples < 2:
            raise ValueError("num_samples должен быть >= 2")

        times = np.linspace(self.t_start, self.t_end, num_samples)
        points = self.evaluate(times)
        return times, points


def _create_open_uniform_knots(
    num_control_points: int,
    degree: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> np.ndarray:
    """Создаёт открытый равномерный узловой вектор в интервале [t_start, t_end]."""
    num_internal_knots = num_control_points - degree - 1

    # Внутренние узлы распределяем равномерно между t_start и t_end
    internal_knots = np.linspace(t_start, t_end, num_internal_knots + 2)[1:-1]

    knots = np.concatenate([
        np.full(degree + 1, t_start),
        internal_knots,
        np.full(degree + 1, t_end),
    ])
    return knots


def _second_difference_matrix(num_control_points: int) -> np.ndarray:
    """Матрица вторых разностей контрольных точек: штраф за изгиб сплайна."""
    return np.diff(np.eye(num_control_points), n=2, axis=0)


def fit_initial_bspline(
    times: np.ndarray,
    points_3d: np.ndarray,
    num_control_points: int | None = None,
    degree: int = 3,
    knot_spacing: float | None = None,
    weights: np.ndarray | None = None,
    smoothing: float = 1e-4,
    loss: str = "linear",
    f_scale: float = 1.0,
    max_iterations: int = 20,
) -> BSplineTrajectory:
    """
    Строит начальный B-сплайн по триангулированным 3D-точкам.

    Узлы открытые равномерные на [times[0], times[-1]]. Их число задаётся либо
    num_control_points, либо knot_spacing (шаг между узлами в единицах times);
    если не задано ни то ни другое, контрольных точек ceil(sqrt(N)).

    Контрольные точки — решение взвешенной задачи наименьших квадратов со штрафом
    на вторые разности контрольных точек (P-сплайн):
        min sum_i w_i |B_i c - p_i|^2 + lambda |D2 c|^2,
        lambda = smoothing * среднее диагонали B^T W B.
    Штраф делает систему невырожденной, когда в интервал узлов не попала ни одна
    точка (участки без наблюдений): сплайн проходит такой участок плавно,
    а не уходит в начало координат, как давал бы псевдообратный lstsq.

    weights — веса точек (например, уверенность или число камер), >= 0.
    loss = "huber" включает робастную подгонку (IRLS): точки с невязкой больше
    f_scale (в единицах мира) получают вес f_scale / невязка.
    """
    times = np.asarray(times, dtype=float)
    points_3d = np.asarray(points_3d, dtype=float)

    if times.ndim != 1 or points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("Некорректные размерности входных массивов")

    if len(times) != len(points_3d) or len(times) < 2:
        raise ValueError("Несоответствие размеров или слишком мало точек")

    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(points_3d)):
        raise ValueError("Входные данные содержат NaN или inf")

    if np.any(np.diff(times) <= 0):
        raise ValueError("times должны строго возрастать.")

    if weights is None:
        weights = np.ones(len(times))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != times.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights должны быть конечными, неотрицательными и по одному на точку")
    if not np.any(weights > 0):
        raise ValueError("Все веса нулевые")

    if loss not in ("linear", "huber"):
        raise ValueError("loss должен быть 'linear' или 'huber'")
    if f_scale <= 0 or smoothing < 0:
        raise ValueError("f_scale должен быть > 0, smoothing >= 0")

    num_points = len(points_3d)
    degree = min(degree, num_points - 1)

    if degree < 1:
        raise ValueError("Степень сплайна должна быть >= 1")

    t_start = float(times[0])
    t_end = float(times[-1])

    if num_control_points is not None and knot_spacing is not None:
        raise ValueError("Задайте либо num_control_points, либо knot_spacing")

    if knot_spacing is not None:
        if knot_spacing <= 0:
            raise ValueError("knot_spacing должен быть > 0")
        num_intervals = max(1, int(np.ceil((t_end - t_start) / knot_spacing)))
        num_control_points = num_intervals + degree

    if num_control_points is None:
        num_control_points = int(np.ceil(np.sqrt(num_points)))
        num_control_points = max(degree + 1, num_control_points)
        num_control_points = min(num_points, num_control_points)

    # Больше контрольных точек, чем данных, допустимо только со сглаживанием
    max_control_points = num_points if smoothing == 0 else np.inf
    if not degree + 1 <= num_control_points <= max_control_points:
        raise ValueError("Некорректное количество контрольных точек")

    # Создаём узлы сразу в физическом времени!
    knots = _create_open_uniform_knots(
        num_control_points, degree, t_start, t_end
    )

    # Матрица дизайна строится по реальному времени times
    basis_matrix = BSpline.design_matrix(
        times, knots, degree, extrapolate=False
    ).toarray()

    penalty = _second_difference_matrix(num_control_points)
    penalty = penalty.T @ penalty

    robust_weights = np.ones(num_points)
    control_points = None
    for _ in range(max_iterations if loss == "huber" else 1):
        w = weights * robust_weights
        normal = basis_matrix.T @ (w[:, None] * basis_matrix)
        lam = smoothing * np.mean(np.diag(normal))
        new_control_points = np.linalg.lstsq(
            normal + lam * penalty, basis_matrix.T @ (w[:, None] * points_3d), rcond=None
        )[0]

        converged = control_points is not None and np.allclose(
            new_control_points, control_points, rtol=0, atol=1e-6 * f_scale
        )
        control_points = new_control_points
        if loss == "linear" or converged:
            break

        residuals = np.linalg.norm(basis_matrix @ control_points - points_3d, axis=1)
        robust_weights = np.minimum(1.0, f_scale / np.maximum(residuals, 1e-12))

    return BSplineTrajectory(
        knots=knots,
        control_points=control_points,
        degree=degree,
        t_start=t_start,
        t_end=t_end,
    )
