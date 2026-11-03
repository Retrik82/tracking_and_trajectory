
from dataclasses import dataclass

import numpy as np

from reconstruction.bspline import BSplineTrajectory
from reconstruction.objective import (
    Cameras,
    ObjectiveData,
    ObservationGeometry,
    observation_cameras,
    validate_objective_data,
)


@dataclass
class TrajectoryMetrics:
    """
    Метрики качества одной траектории.

    Ошибки репроекции — в пикселях. Ускорение и рывок — в единицах мира
    за (единицу времени сплайна)^2 и ^3, например см/кадр^2.
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
    cameras: Cameras,
    data: ObjectiveData,
) -> np.ndarray:
    """
    Вычисляет евклидовы ошибки репроекции для всех наблюдений.

    Возвращает массив (M,), где каждый элемент — ошибка в пикселях;
    inf, если точка сплайна оказалась позади камеры.
    """

    validate_objective_data(data, cameras, trajectory)
    geometry = ObservationGeometry.from_cameras(
        observation_cameras(cameras, data)
    )

    predicted_points = trajectory.evaluate(data.times)
    pixels, depth = geometry.project(predicted_points)

    errors = np.linalg.norm(pixels - data.pixels, axis=1)
    errors[depth <= 0] = np.inf

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

    Среднеквадратичные значения по времени (интеграл / длительность).
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

    # Сплайн задан в физическом времени — производные берутся напрямую.
    acceleration = trajectory.derivative(2).evaluate(times)
    jerk = trajectory.derivative(3).evaluate(times)

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
    cameras: Cameras,
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

    for camera_index in np.unique(data.camera_indices).tolist():
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


def position_error_stats(
    trajectory: BSplineTrajectory,
    times: np.ndarray,
    target_points: np.ndarray,
    units_per_meter: float = 100.0,
) -> dict[str, float]:
    """
    Ошибка положения сплайна относительно эталона (CSV цели), в метрах.

    Строки target_points с NaN (испорченные ячейки CSV) пропускаются.
    """

    times = np.asarray(times, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    if target_points.shape != (len(times), 3):
        raise ValueError("target_points должен иметь форму (len(times), 3)")

    valid = np.all(np.isfinite(target_points), axis=1)

    if not np.any(valid):
        raise ValueError("Нет корректных эталонных точек")

    errors = np.linalg.norm(
        trajectory.evaluate(times[valid]) - target_points[valid],
        axis=1,
    ) / units_per_meter

    return {
        "count": int(len(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max": float(np.max(errors)),
    }


def compare_trajectories(
    initial_trajectory: BSplineTrajectory,
    optimized_trajectory: BSplineTrajectory,
    cameras: Cameras,
    data: ObjectiveData,
    target_times: np.ndarray | None = None,
    target_points: np.ndarray | None = None,
    units_per_meter: float = 100.0,
) -> dict[str, dict[str, float]]:
    """
    Сравнивает начальную и оптимизированную траектории.

    Возвращает словарь с метриками обеих траекторий. Если передан эталон
    (target_times, target_points), добавляет ошибку положения в метрах.
    """

    result = {}

    for name, trajectory in (
        ("initial", initial_trajectory),
        ("optimized", optimized_trajectory),
    ):
        metrics = calculate_metrics(
            trajectory,
            cameras,
            data,
        )

        result[name] = {
            "reprojection_mae": metrics.reprojection_mae,
            "reprojection_rmse": metrics.reprojection_rmse,
            "reprojection_max": metrics.reprojection_max,
            "acceleration_rms": metrics.acceleration_rms,
            "jerk_rms": metrics.jerk_rms,
        }

        if target_times is not None and target_points is not None:
            position = position_error_stats(
                trajectory,
                target_times,
                target_points,
                units_per_meter,
            )

            result[name].update({
                f"position_{key}_m" if key != "count" else "position_count": value
                for key, value in position.items()
            })

    return result
