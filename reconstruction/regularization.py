
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


def _derivative_residuals(
    trajectory: BSplineTrajectory,
    times: np.ndarray,
    order: int,
    weight: float,
) -> np.ndarray:
    if weight < 0:
        raise ValueError("weight не может быть отрицательным")

    if trajectory.degree < order:
        raise ValueError(
            f"Для производной порядка {order} нужна степень сплайна >= {order}"
        )

    times = np.asarray(times, dtype=float)

    weights = trapezoidal_weights(times)

    # BSplineTrajectory задан в физическом времени (узлы в тех же единицах,
    # что и times), поэтому производная берётся напрямую, без нормализации.
    derivative = trajectory.derivative(order).evaluate(times)

    return (
        np.sqrt(weight * weights)[:, None]
        * derivative
    )


def acceleration_residuals(
    trajectory: BSplineTrajectory,
    times: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """
    Возвращает остатки регуляризации по ускорению.

    Минимизируется приближение к:

        weight * integral(||P''(t)||^2 dt)

    Единицы P'' — единицы мира / (единица времени сплайна)^2,
    например см / кадр^2. Результат имеет размер (N, 3).
    """

    return _derivative_residuals(trajectory, times, 2, weight)


def jerk_residuals(
    trajectory: BSplineTrajectory,
    times: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """
    Возвращает остатки регуляризации по рывку.

    Минимизируется приближение к:

        weight * integral(||P'''(t)||^2 dt)

    Единицы P''' — единицы мира / (единица времени сплайна)^3.
    Результат имеет размер (N, 3).
    """

    return _derivative_residuals(trajectory, times, 3, weight)


def regularization_residuals(
    trajectory: BSplineTrajectory,
    times: np.ndarray,
    lambda_acceleration: float = 10.0,
    lambda_jerk: float = 100.0,
) -> np.ndarray:
    """
    Объединяет остатки по ускорению и рывку.

    Возвращает одномерный вектор для least_squares:
    сначала N*3 остатков ускорения, затем N*3 остатков рывка.
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
