
import numpy as np

from reconstruction.bspline import BSplineTrajectory


def trapezoidal_weights(times: np.ndarray) -> np.ndarray:
    """
    Вычисляет веса численного интегрирования методом трапеций.

    Для равномерной сетки внутренние веса равны dt,
    крайние — dt / 2.
    """

    times = np.asarray(times, dtype=float)

    if times.ndim != 1 or len(times) < 2:
        raise ValueError("Нужно минимум два момента времени")

    if np.any(np.diff(times) <= 0):
        raise ValueError("times должны строго возрастать")

    weights = np.empty_like(times)

    weights[0] = (times[1] - times[0]) / 2
    weights[-1] = (times[-1] - times[-2]) / 2

    if len(times) > 2:
        weights[1:-1] = (
            times[2:] - times[:-2]
        ) / 2

    return weights


def acceleration_residuals(
    trajectory: BSplineTrajectory,
    times: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """
    Возвращает остатки регуляризации по ускорению.

    Минимизируется приближение к:

        weight * integral(||P''(t)||^2 dt)

    Результат имеет размер (N, 3).
    """

    if weight < 0:
        raise ValueError("weight не может быть отрицательным")

    if trajectory.degree < 2:
        raise ValueError(
            "Для ускорения нужна степень сплайна >= 2"
        )

    times = np.asarray(times, dtype=float)

    weights = trapezoidal_weights(times)

    # Производная B-сплайна второго порядка.
    derivative_spline = trajectory._spline.derivative(2)

    # В BSpline время нормализовано в [0, 1].
    # Поэтому для перехода к физическому времени
    # делим вторую производную на duration^2.
    duration = trajectory.t_end - trajectory.t_start

    acceleration = derivative_spline(
        (times - trajectory.t_start) / duration
    ) / duration**2

    return (
        np.sqrt(weight * weights)[:, None]
        * acceleration
    )


def jerk_residuals(
    trajectory: BSplineTrajectory,
    times: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """
    Возвращает остатки регуляризации по рывку.

    Минимизируется приближение к:

        weight * integral(||P'''(t)||^2 dt)

    Результат имеет размер (N, 3).
    """

    if weight < 0:
        raise ValueError("weight не может быть отрицательным")

    if trajectory.degree < 3:
        raise ValueError(
            "Для рывка нужна степень сплайна >= 3"
        )

    times = np.asarray(times, dtype=float)

    weights = trapezoidal_weights(times)

    derivative_spline = trajectory._spline.derivative(3)

    duration = trajectory.t_end - trajectory.t_start

    jerk = derivative_spline(
        (times - trajectory.t_start) / duration
    ) / duration**3

    return (
        np.sqrt(weight * weights)[:, None]
        * jerk
    )


def regularization_residuals(
    trajectory: BSplineTrajectory,
    times: np.ndarray,
    lambda_acceleration: float = 1e-3,
    lambda_jerk: float = 1e-4,
) -> np.ndarray:
    """
    Объединяет остатки по ускорению и рывку.

    Возвращает одномерный вектор для least_squares.
    """

    acceleration = acceleration_residuals(
        trajectory,
        times,
        weight=lambda_acceleration,
    )

    jerk = jerk_residuals(
        trajectory,
        times,
        weight=lambda_jerk,
    )

    return np.concatenate([
        acceleration.ravel(),
        jerk.ravel(),
    ])
