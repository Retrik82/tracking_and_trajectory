
from dataclasses import dataclass

import numpy as np

from mct.reconstruction.bspline import BSplineTrajectory
from mct.reconstruction.camera import Camera
from mct.reconstruction.objective import (
    ObjectiveData,
    project_point,
)


@dataclass
class TrajectoryMetrics:
    """
    Метрики качества одной траектории.

    Все ошибки репроекции измеряются в пикселях.
    """

    reprojection_mae: float
    reprojection_rmse: float
    reprojection_max: float

    acceleration_rms: float
    jerk_rms: float

    reprojection_errors: np.ndarray
    errors_by_camera: dict[int, dict[str, float]]


def calculate_reprojection_errors(
    trajectory: BSplineTrajectory,
    cameras: list[Camera],
    data: ObjectiveData,
) -> np.ndarray:
    """
    Вычисляет евклидовы ошибки репроекции для всех наблюдений.

    Возвращает массив (M,), где каждый элемент — ошибка в пикселях.
    """

    predicted_points = trajectory.evaluate(data.times)

    errors = np.zeros(len(data.times), dtype=float)

    for index, point_3d in enumerate(predicted_points):
        camera_index = int(data.camera_indices[index])

        camera = cameras[camera_index]

        predicted_pixel, depth = project_point(
            camera,
            point_3d,
        )

        observed_pixel = data.pixels[index]

        if depth <= 0:
            errors[index] = np.inf
            continue

        errors[index] = np.linalg.norm(
            predicted_pixel - observed_pixel
        )

    return errors


def calculate_smoothness(
    trajectory: BSplineTrajectory,
    num_samples: int = 500,
) -> tuple[float, float]:
    """
    Вычисляет RMS ускорения и RMS рывка.

    Ускорение:
        P''(t)

    Рывок:
        P'''(t)

    Производные учитывают переход от нормализованного
    времени сплайна к исходному времени.
    """

    if trajectory.degree < 3:
        raise ValueError(
            "Для вычисления рывка нужна степень сплайна >= 3"
        )

    if num_samples < 2:
        raise ValueError("num_samples должен быть >= 2")

    times = np.linspace(
        trajectory.t_start,
        trajectory.t_end,
        num_samples,
    )

    duration = trajectory.t_end - trajectory.t_start

    normalized_times = (
        (times - trajectory.t_start) / duration
    )

    acceleration_spline = trajectory._spline.derivative(2)
    jerk_spline = trajectory._spline.derivative(3)

    acceleration = (
        acceleration_spline(normalized_times) / duration**2
    )

    jerk = (
        jerk_spline(normalized_times) / duration**3
    )

    acceleration_squared = np.sum(
        acceleration**2,
        axis=1,
    )

    jerk_squared = np.sum(
        jerk**2,
        axis=1,
    )

    acceleration_rms = np.sqrt(
        np.trapezoid(
            acceleration_squared,
            times,
        ) / duration
    )

    jerk_rms = np.sqrt(
        np.trapezoid(
            jerk_squared,
            times,
        ) / duration
    )

    return (
        float(acceleration_rms),
        float(jerk_rms),
    )


def calculate_metrics(
    trajectory: BSplineTrajectory,
    cameras: list[Camera],
    data: ObjectiveData,
    smoothness_samples: int = 500,
) -> TrajectoryMetrics:
    """
    Вычисляет полный набор метрик для одной траектории.
    """

    errors = calculate_reprojection_errors(
        trajectory,
        cameras,
        data,
    )

    finite_errors = errors[np.isfinite(errors)]

    if len(finite_errors) == 0:
        raise ValueError(
            "Не удалось получить корректные репроекционные ошибки"
        )

    mae = float(np.mean(finite_errors))

    rmse = float(
        np.sqrt(np.mean(finite_errors**2))
    )

    max_error = float(np.max(finite_errors))

    acceleration_rms, jerk_rms = calculate_smoothness(
        trajectory,
        num_samples=smoothness_samples,
    )

    # Отдельные показатели для каждой камеры.
    errors_by_camera = {}

    for camera_index in range(len(cameras)):
        mask = data.camera_indices == camera_index

        camera_errors = errors[mask]
        camera_errors = camera_errors[
            np.isfinite(camera_errors)
        ]

        if len(camera_errors) == 0:
            continue

        errors_by_camera[camera_index] = {
            "count": int(len(camera_errors)),
            "mae": float(np.mean(camera_errors)),
            "rmse": float(
                np.sqrt(np.mean(camera_errors**2))
            ),
            "max": float(np.max(camera_errors)),
        }

    return TrajectoryMetrics(
        reprojection_mae=mae,
        reprojection_rmse=rmse,
        reprojection_max=max_error,
        acceleration_rms=acceleration_rms,
        jerk_rms=jerk_rms,
        reprojection_errors=errors,
        errors_by_camera=errors_by_camera,
    )


def compare_trajectories(
    initial_trajectory: BSplineTrajectory,
    optimized_trajectory: BSplineTrajectory,
    cameras: list[Camera],
    data: ObjectiveData,
) -> dict[str, dict[str, float]]:
    """
    Сравнивает начальную и оптимизированную траектории.

    Возвращает словарь с метриками обеих траекторий.
    """

    initial = calculate_metrics(
        initial_trajectory,
        cameras,
        data,
    )

    optimized = calculate_metrics(
        optimized_trajectory,
        cameras,
        data,
    )

    return {
        "initial": {
            "reprojection_mae": initial.reprojection_mae,
            "reprojection_rmse": initial.reprojection_rmse,
            "reprojection_max": initial.reprojection_max,
            "acceleration_rms": initial.acceleration_rms,
            "jerk_rms": initial.jerk_rms,
        },
        "optimized": {
            "reprojection_mae": optimized.reprojection_mae,
            "reprojection_rmse": optimized.reprojection_rmse,
            "reprojection_max": optimized.reprojection_max,
            "acceleration_rms": optimized.acceleration_rms,
            "jerk_rms": optimized.jerk_rms,
        },
    }
