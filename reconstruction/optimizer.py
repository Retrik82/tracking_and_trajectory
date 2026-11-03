
from dataclasses import dataclass

from scipy.optimize import least_squares

from reconstruction.bspline import BSplineTrajectory
from reconstruction.objective import (
    Cameras,
    ObjectiveData,
    TrajectoryObjective,
    build_trajectory,
)


@dataclass
class OptimizationResult:
    """
    Результат оптимизации.

    Attributes:
        trajectory:
            Оптимизированная B-сплайн траектория.

        scipy_result:
            Полный результат scipy.optimize.least_squares.

        initial_error:
            Ошибка до оптимизации.

        final_error:
            Ошибка после оптимизации.
    """

    trajectory: BSplineTrajectory
    scipy_result: object
    initial_error: dict[str, float]
    final_error: dict[str, float]


def optimize_trajectory(
    initial_trajectory: BSplineTrajectory,
    cameras: Cameras,
    data: ObjectiveData,
    lambda_acceleration: float = 10.0,
    lambda_jerk: float = 100.0,
    regularization_samples: int = 200,
    max_nfev: int = 1000,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    loss: str = "huber",
    f_scale: float = 5.0,
) -> OptimizationResult:
    """
    Оптимизирует контрольные точки B-сплайна.

    Минимизируется:

        E = E_reprojection
          + lambda_a * E_acceleration
          + lambda_j * E_jerk

    Args:
        initial_trajectory:
            Начальная B-сплайн траектория.

        cameras:
            Список неподвижных камер (по camera_indices) или функция
            (индекс камеры, кадр CSV) -> Camera, если камера поворачивается.

        data:
            Исходные пиксельные наблюдения.

        lambda_acceleration:
            Вес регуляризации по ускорению.

        lambda_jerk:
            Вес регуляризации по рывку.

            Веса зависят от единиц: значения по умолчанию подобраны для мира
            в сантиметрах и времени в кадрах (30 FPS). На детекциях YOLO при
            1e-3 / 1e-4 регуляризация почти не действует (максимум скорости
            ~50 м/с при реальных ~32 м/с, оптимизатор не сходится за 2000 итераций),
            при 10 / 100 сходится и даёт ~40 м/с при той же ошибке перепроецирования.

        regularization_samples:
            Количество временных точек для регуляризации.

        max_nfev:
            Максимальное число вычислений функции остатков.

        ftol, xtol, gtol:
            Критерии остановки SciPy.

        loss:
            Тип функции потерь: linear, soft_l1, huber,
            cauchy или arctan. По умолчанию huber: детекции содержат выбросы.

        f_scale:
            Масштаб для робастной функции потерь, пиксели (с учётом sqrt(confidence)).

    Returns:
        OptimizationResult.
    """

    if initial_trajectory.degree < 3:
        raise ValueError(
            "Для регуляризации по рывку нужна степень сплайна >= 3"
        )

    if max_nfev <= 0:
        raise ValueError("max_nfev должен быть положительным")

    if f_scale <= 0:
        raise ValueError("f_scale должен быть положительным")

    # Создаём объект общей функции ошибки.
    objective = TrajectoryObjective(
        template=initial_trajectory,
        cameras=cameras,
        data=data,
        lambda_acceleration=lambda_acceleration,
        lambda_jerk=lambda_jerk,
        regularization_samples=regularization_samples,
    )

    # Начальные контрольные точки.
    initial_control_points = (
        initial_trajectory.control_points.copy()
    )

    # Преобразуем матрицу (N, 3) в вектор (3N,).
    initial_parameters = initial_control_points.ravel()

    # Считаем начальную ошибку.
    initial_error = objective.error_components(
        initial_parameters
    )

    # Запускаем нелинейную оптимизацию.
    scipy_result = least_squares(
        fun=objective.residuals,
        x0=initial_parameters,
        method="trf",
        jac="2-point",
        jac_sparsity=objective.jac_sparsity(),
        x_scale="jac",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        verbose=0,
    )

    # Восстанавливаем матрицу контрольных точек.
    optimized_control_points = scipy_result.x.reshape(
        -1,
        3,
    )

    # Создаём оптимизированную траекторию.
    optimized_trajectory = build_trajectory(
        template=initial_trajectory,
        control_points=optimized_control_points,
    )

    # Считаем итоговую ошибку.
    final_error = objective.error_components(
        scipy_result.x
    )

    return OptimizationResult(
        trajectory=optimized_trajectory,
        scipy_result=scipy_result,
        initial_error=initial_error,
        final_error=final_error,
    )
