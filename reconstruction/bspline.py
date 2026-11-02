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


def fit_initial_bspline(
    times: np.ndarray,
    points_3d: np.ndarray,
    num_control_points: int | None = None,
    degree: int = 3,
) -> BSplineTrajectory:
    """Строит начальный B-сплайн по триангулированным 3D-точкам."""
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

    num_points = len(points_3d)
    degree = min(degree, num_points - 1)

    if degree < 1:
        raise ValueError("Степень сплайна должна быть >= 1")

    if num_control_points is None:
        num_control_points = int(np.ceil(np.sqrt(num_points)))
        num_control_points = max(degree + 1, num_control_points)
        num_control_points = min(num_points, num_control_points)

    if not degree + 1 <= num_control_points <= num_points:
        raise ValueError("Некорректное количество контрольных точек")

    t_start = float(times[0])
    t_end = float(times[-1])

    # Создаём узлы сразу в физическом времени!
    knots = _create_open_uniform_knots(
        num_control_points, degree, t_start, t_end
    )

    # Матрица дизайна строится по реальному времени times
    basis_matrix = BSpline.design_matrix(
        times, knots, degree, extrapolate=False
    ).toarray()

    control_points, _, _, _ = np.linalg.lstsq(
        basis_matrix, points_3d, rcond=None
    )

    return BSplineTrajectory(
        knots=knots,
        control_points=control_points,
        degree=degree,
        t_start=t_start,
        t_end=t_end,
    )
