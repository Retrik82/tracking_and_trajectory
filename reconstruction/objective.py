
from dataclasses import dataclass

import numpy as np

from mct.reconstruction.bspline import BSplineTrajectory
from mct.reconstruction.camera import Camera
from mct.reconstruction.regularization import (
    regularization_residuals,
)


@dataclass
class ObjectiveData:
    """
    Данные, необходимые для вычисления функции ошибки.

    Attributes:
        times:
            Времена наблюдений, shape (M,).

        camera_indices:
            Индекс камеры для каждого наблюдения, shape (M,).

        pixels:
            Пиксельные координаты (u, v), shape (M, 2).

        confidence:
            Уверенность детекций, shape (M,).
    """

    times: np.ndarray
    camera_indices: np.ndarray
    pixels: np.ndarray
    confidence: np.ndarray


def project_point(
    camera: Camera,
    point_3d: np.ndarray,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """
    Проецирует 3D-точку в изображение камеры.

    Используется модель:

        P_camera = R @ (P_world - C)

        u = fx * Xc / Zc + cx
        v = fy * Yc / Zc + cy

    Returns:
        pixel:
            Проекция (u, v).

        depth:
            Координата Zc точки в системе камеры.
    """

    point_camera = camera.R @ (
        point_3d - camera.position
    )

    x_camera, y_camera, z_camera = point_camera

    # Не допускаем деление на ноль.
    safe_depth = max(z_camera, min_depth)

    u = camera.fx * x_camera / safe_depth + camera.cx
    v = camera.fy * y_camera / safe_depth + camera.cy

    return np.array([u, v]), float(z_camera)


def build_trajectory(
    template: BSplineTrajectory,
    control_points: np.ndarray,
) -> BSplineTrajectory:
    """
    Создаёт новую траекторию с заданными контрольными точками.

    Узловой вектор и остальные параметры остаются неизменными.
    """

    return BSplineTrajectory(
        knots=template.knots.copy(),
        control_points=np.asarray(
            control_points,
            dtype=float,
        ).copy(),
        degree=template.degree,
        t_start=template.t_start,
        t_end=template.t_end,
    )


def validate_objective_data(
    data: ObjectiveData,
    cameras: list[Camera],
    trajectory: BSplineTrajectory,
) -> None:
    """Проверяет входные данные функции ошибки."""

    data.times = np.asarray(data.times, dtype=float)
    data.camera_indices = np.asarray(
        data.camera_indices,
        dtype=int,
    )
    data.pixels = np.asarray(data.pixels, dtype=float)
    data.confidence = np.asarray(
        data.confidence,
        dtype=float,
    )

    count = len(data.times)

    if data.times.ndim != 1:
        raise ValueError("times должен быть одномерным")

    if data.camera_indices.shape != (count,):
        raise ValueError("Некорректная форма camera_indices")

    if data.pixels.shape != (count, 2):
        raise ValueError("pixels должен иметь форму (M, 2)")

    if data.confidence.shape != (count,):
        raise ValueError("confidence должен иметь форму (M,)")

    if count == 0:
        raise ValueError("Нет наблюдений для оптимизации")

    if not np.all(np.isfinite(data.times)):
        raise ValueError("times содержит NaN или inf")

    if not np.all(np.isfinite(data.pixels)):
        raise ValueError("pixels содержит NaN или inf")

    if not np.all(np.isfinite(data.confidence)):
        raise ValueError("confidence содержит NaN или inf")

    if np.any(data.camera_indices < 0):
        raise ValueError("Индекс камеры не может быть отрицательным")

    if np.any(data.camera_indices >= len(cameras)):
        raise ValueError("Указан несуществующий индекс камеры")

    if np.any(data.confidence < 0):
        raise ValueError("confidence не может быть отрицательным")

    if np.any(data.times < trajectory.t_start) or np.any(
        data.times > trajectory.t_end
    ):
        raise ValueError(
            "Время наблюдения находится вне интервала траектории"
        )


class TrajectoryObjective:
    """
    Общая функция ошибки для оптимизации B-сплайна.

    Минимизируется:

        E = E_reprojection
          + lambda_a * E_acceleration
          + lambda_j * E_jerk
    """

    def __init__(
        self,
        template: BSplineTrajectory,
        cameras: list[Camera],
        data: ObjectiveData,
        lambda_acceleration: float = 1e-3,
        lambda_jerk: float = 1e-4,
        regularization_samples: int = 200,
        min_depth: float = 1e-4,
        depth_penalty_scale: float = 100.0,
    ):
        if lambda_acceleration < 0 or lambda_jerk < 0:
            raise ValueError("Коэффициенты регуляризации не могут быть отрицательными")

        if regularization_samples < 2:
            raise ValueError("regularization_samples должен быть >= 2")

        if min_depth <= 0:
            raise ValueError("min_depth должен быть положительным")

        self.template = template
        self.cameras = cameras
        self.data = data

        self.lambda_acceleration = lambda_acceleration
        self.lambda_jerk = lambda_jerk

        self.min_depth = min_depth
        self.depth_penalty_scale = depth_penalty_scale

        validate_objective_data(
            self.data,
            self.cameras,
            self.template,
        )

        self.regularization_times = np.linspace(
            template.t_start,
            template.t_end,
            regularization_samples,
        )

    def reprojection_residuals(
        self,
        trajectory: BSplineTrajectory,
    ) -> np.ndarray:
        """
        Вычисляет пиксельные остатки между детекциями
        и проекциями B-сплайна.

        Возвращает вектор длиной 2 * M + M:
        две координаты ошибки и один штраф за глубину
        на каждое наблюдение.
        """

        predicted_points = trajectory.evaluate(
            self.data.times
        )

        residuals = []
        depth_residuals = []

        for index, point_3d in enumerate(predicted_points):
            camera_index = self.data.camera_indices[index]
            camera = self.cameras[camera_index]

            observed_pixel = self.data.pixels[index]

            predicted_pixel, depth = project_point(
                camera,
                point_3d,
                min_depth=self.min_depth,
            )

            confidence_weight = np.sqrt(
                self.data.confidence[index]
            )

            pixel_error = (
                predicted_pixel - observed_pixel
            ) * confidence_weight

            residuals.extend(pixel_error)

            # Точка должна находиться перед камерой.
            # Если depth меньше min_depth, добавляем штраф.
            depth_error = max(
                self.min_depth - depth,
                0.0,
            ) * self.depth_penalty_scale

            depth_residuals.append(depth_error)

        return np.concatenate([
            np.asarray(residuals),
            np.asarray(depth_residuals),
        ])

    def regularization_residuals(
        self,
        trajectory: BSplineTrajectory,
    ) -> np.ndarray:
        """Вычисляет штрафы по ускорению и рывку."""

        return regularization_residuals(
            trajectory=trajectory,
            times=self.regularization_times,
            lambda_acceleration=self.lambda_acceleration,
            lambda_jerk=self.lambda_jerk,
        )

    def residuals(
        self,
        flat_control_points: np.ndarray,
    ) -> np.ndarray:
        """
        Главная функция для scipy.optimize.least_squares.

        На вход получает одномерный массив контрольных точек.
        На выходе возвращает одномерный вектор остатков.
        """

        control_points = np.asarray(
            flat_control_points,
            dtype=float,
        ).reshape(-1, 3)

        trajectory = build_trajectory(
            self.template,
            control_points,
        )

        reprojection = self.reprojection_residuals(
            trajectory
        )

        regularization = self.regularization_residuals(
            trajectory
        )

        return np.concatenate([
            reprojection,
            regularization,
        ])

    def error_components(
        self,
        flat_control_points: np.ndarray,
    ) -> dict[str, float]:
        """
        Возвращает отдельные составляющие ошибки.

        Полезно для диагностики оптимизации.
        """

        control_points = np.asarray(
            flat_control_points,
            dtype=float,
        ).reshape(-1, 3)

        trajectory = build_trajectory(
            self.template,
            control_points,
        )

        reprojection = self.reprojection_residuals(
            trajectory
        )

        regularization = self.regularization_residuals(
            trajectory
        )

        # В reprojection_residuals последние M элементов —
        # штрафы за нахождение точек позади камеры.
        count = len(self.data.times)

        pixel_residuals = reprojection[:2 * count]
        depth_residuals = reprojection[2 * count:]

        return {
            "reprojection": float(
                np.sum(pixel_residuals**2)
            ),
            "depth_penalty": float(
                np.sum(depth_residuals**2)
            ),
            "regularization": float(
                np.sum(regularization**2)
            ),
            "total": float(
                np.sum(reprojection**2)
                + np.sum(regularization**2)
            ),
        }
