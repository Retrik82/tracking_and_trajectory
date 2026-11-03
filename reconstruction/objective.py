
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix

from reconstruction.bspline import BSplineTrajectory
from reconstruction.camera import Camera
from reconstruction.regularization import (
    regularization_residuals,
)

# Камера для наблюдения: (индекс камеры, номер кадра CSV) -> Camera с позой этого кадра.
# Нужна для камер, которые поворачиваются (Camera5 меняет yaw на кадрах 565–960).
CameraProvider = Callable[[int, int], Camera]
Cameras = Sequence[Camera] | CameraProvider


@dataclass
class ObjectiveData:
    """
    Данные, необходимые для вычисления функции ошибки.

    Attributes:
        times:
            Времена наблюдений в единицах времени сплайна, shape (M,).

        camera_indices:
            Индекс камеры для каждого наблюдения, shape (M,).

        pixels:
            Пиксельные координаты (u, v), shape (M, 2).

        confidence:
            Уверенность детекций, shape (M,).

        frames:
            Номер кадра CSV, из строки которого берётся поза камеры, shape (M,).
            По умолчанию — times, округлённые до целого. Используется, только
            если камеры заданы функцией CameraProvider.
    """

    times: np.ndarray
    camera_indices: np.ndarray
    pixels: np.ndarray
    confidence: np.ndarray
    frames: np.ndarray | None = None


def project_point(
    camera: Camera,
    point_3d: np.ndarray,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """
    Проецирует 3D-точку в изображение камеры.

    Используется модель камеры проекта (camera.R — поворот мир -> камера,
    строки right, -up, forward):

        P_camera = R @ (P_world - C)

        u = f_x * Xc / Zc + c_x
        v = f_y * Yc / Zc + c_y

    Returns:
        pixel:
            Проекция (u, v).

        depth:
            Координата Zc точки в системе камеры.
    """

    point_camera = camera.R @ (
        np.asarray(point_3d, dtype=float) - camera.position
    )

    x_camera, y_camera, z_camera = point_camera

    # Не допускаем деление на ноль.
    safe_depth = max(z_camera, min_depth)

    u = camera.f_x * x_camera / safe_depth + camera.c_x
    v = camera.f_y * y_camera / safe_depth + camera.c_y

    return np.array([u, v]), float(z_camera)


@dataclass
class ObservationGeometry:
    """Параметры камеры для каждого наблюдения, собранные в массивы для векторной проекции."""

    rotation: np.ndarray      # (M, 3, 3)
    position: np.ndarray      # (M, 3)
    focal: np.ndarray         # (M, 2) f_x, f_y
    center: np.ndarray        # (M, 2) c_x, c_y

    @classmethod
    def from_cameras(cls, cameras: Sequence[Camera]) -> "ObservationGeometry":
        return cls(
            rotation=np.stack([c.R for c in cameras]),
            position=np.stack([c.position for c in cameras]),
            focal=np.array([[c.f_x, c.f_y] for c in cameras], dtype=float),
            center=np.array([[c.c_x, c.c_y] for c in cameras], dtype=float),
        )

    def project(self, points: np.ndarray, min_depth: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Проекция точек (M, 3), по одной на наблюдение. Возвращает пиксели (M, 2) и глубину (M,)."""
        cam = np.einsum("mij,mj->mi", self.rotation, points - self.position)
        depth = cam[:, 2]
        safe = np.maximum(depth, min_depth)[:, None]
        pixels = self.focal * cam[:, :2] / safe + self.center
        return pixels, depth


def observation_cameras(cameras: Cameras, data: ObjectiveData) -> list[Camera]:
    """
    Камера для каждого наблюдения.

    Список Camera — неподвижные камеры, выбираются по camera_indices.
    CameraProvider — поза берётся из CSV на кадре data.frames (повороты камеры учитываются).
    """
    if callable(cameras) and not isinstance(cameras, Camera):
        cache: dict[tuple[int, int], Camera] = {}
        result = []
        for index, frame in zip(data.camera_indices, data.frames):
            key = (int(index), int(frame))
            if key not in cache:
                cache[key] = cameras(*key)
            result.append(cache[key])
        return result
    return [cameras[int(index)] for index in data.camera_indices]


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
    cameras: Cameras,
    trajectory: BSplineTrajectory,
) -> None:
    """Проверяет входные данные функции ошибки и приводит массивы к нужным типам."""

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

    if data.frames is None:
        data.frames = np.rint(data.times).astype(int)
    data.frames = np.asarray(data.frames)
    if data.frames.shape != (count,) or not np.issubdtype(data.frames.dtype, np.integer):
        raise ValueError("frames должен быть целочисленным массивом формы (M,)")

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

    if not callable(cameras) and np.any(data.camera_indices >= len(cameras)):
        raise ValueError("Указан несуществующий индекс камеры")

    if np.any(data.confidence < 0):
        raise ValueError("confidence не может быть отрицательным")

    if np.any(data.times < trajectory.t_start) or np.any(
        data.times > trajectory.t_end
    ):
        raise ValueError(
            "Время наблюдения находится вне интервала траектории"
        )


def _support(trajectory: BSplineTrajectory, times: np.ndarray) -> np.ndarray:
    """Индексы контрольных точек, влияющих на сплайн в моменты times: (len(times), degree + 1)."""
    k = trajectory.degree
    n = len(trajectory.control_points)
    interval = np.searchsorted(trajectory.knots, times, side="right") - 1
    interval = np.clip(interval, k, n - 1)
    return interval[:, None] - k + np.arange(k + 1)[None, :]


class TrajectoryObjective:
    """
    Общая функция ошибки для оптимизации B-сплайна.

    Минимизируется:

        E = E_reprojection
          + lambda_a * E_acceleration
          + lambda_j * E_jerk

    Вектор остатков: [2M пиксельных (u, v), M штрафов за глубину,
    3S по ускорению, 3S по рывку], где S = regularization_samples.
    """

    def __init__(
        self,
        template: BSplineTrajectory,
        cameras: Cameras,
        data: ObjectiveData,
        lambda_acceleration: float = 10.0,
        lambda_jerk: float = 100.0,
        regularization_samples: int = 200,
        min_depth: float = 1.0,
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

        # min_depth — в единицах мира; точка ближе к плоскости камеры считается «позади».
        self.min_depth = min_depth
        self.depth_penalty_scale = depth_penalty_scale

        validate_objective_data(
            self.data,
            self.cameras,
            self.template,
        )

        self.geometry = ObservationGeometry.from_cameras(
            observation_cameras(self.cameras, self.data)
        )
        self.sqrt_confidence = np.sqrt(self.data.confidence)

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

        Если точка оказалась позади камеры, пиксельный остаток обнуляется
        (проекция там не имеет смысла и дала бы огромные значения),
        а работает только штраф за глубину.
        """

        predicted_points = trajectory.evaluate(
            self.data.times
        )

        pixels, depth = self.geometry.project(
            predicted_points,
            min_depth=self.min_depth,
        )
        in_front = depth > self.min_depth

        pixel_error = np.where(
            in_front[:, None],
            (pixels - self.data.pixels) * self.sqrt_confidence[:, None],
            0.0,
        )

        depth_error = np.maximum(
            self.min_depth - depth,
            0.0,
        ) * self.depth_penalty_scale

        return np.concatenate([
            pixel_error.ravel(),
            depth_error,
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

    def jac_sparsity(self) -> lil_matrix:
        """
        Структура якобиана: каждый остаток зависит только от degree + 1
        соседних контрольных точек. Без неё least_squares считает плотный
        якобиан конечными разностями — на 1000 кадрах это в десятки раз медленнее.
        """

        count = len(self.data.times)
        samples = len(self.regularization_times)
        num_params = 3 * len(self.template.control_points)
        sparsity = lil_matrix((3 * count + 6 * samples, num_params), dtype=np.int8)

        for i, cps in enumerate(_support(self.template, self.data.times)):
            columns = (3 * cps[:, None] + np.arange(3)).ravel()
            sparsity[2 * i, columns] = 1
            sparsity[2 * i + 1, columns] = 1
            sparsity[2 * count + i, columns] = 1

        offset = 3 * count
        for block in range(2):                      # ускорение, затем рывок
            for s, cps in enumerate(_support(self.template, self.regularization_times)):
                for axis in range(3):
                    sparsity[offset + 3 * s + axis, 3 * cps + axis] = 1
            offset += 3 * samples

        return sparsity

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
