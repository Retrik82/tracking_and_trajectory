from dataclasses import dataclass

import numpy as np

from reconstruction.camera import Camera

# Если система для ближайшей точки к лучам обусловлена хуже этого порога,
# лучи почти параллельны и точка по ним не определяется
MAX_CONDITION_NUMBER = 1e8


def pixel_to_ray(camera: Camera, u: float, v: float) -> np.ndarray:
    """Единичное направление луча через пиксель (u, v) в мировых координатах."""
    pixel = np.array([u, v, 1.0])

    # Пиксель -> координаты камеры
    direction_camera = np.linalg.inv(camera.K) @ pixel

    # Координаты камеры -> мировые координаты (R — поворот мир -> камера)
    direction_world = camera.R.T @ direction_camera

    # нормализация
    return direction_world / np.linalg.norm(direction_world)


def triangulate(cameras: list[Camera], detections: list[tuple[float, float]]) -> np.ndarray:
    """
    Точка, ближайшая ко всем лучам в смысле наименьших квадратов:
    минимизирует сумму квадратов расстояний от точки до каждого луча.
    """
    if len(cameras) != len(detections):
        raise ValueError("Количество камер и детекций не совпадает")

    if len(cameras) < 2:
        raise ValueError("Для триангуляции нужны минимум 2 камеры")

    A_total = np.zeros((3, 3))
    b_total = np.zeros(3)
    I = np.eye(3)

    for camera, (u, v) in zip(cameras, detections):

        # Центр камеры
        C = camera.position

        # Направление луча
        d = pixel_to_ray(camera, u, v)

        # Матрица проекции на плоскость, перпендикулярную лучу
        A = I - np.outer(d, d)

        A_total += A
        b_total += A @ C

    if np.linalg.cond(A_total) > MAX_CONDITION_NUMBER:
        raise np.linalg.LinAlgError("Лучи почти параллельны, точка не определяется")

    return np.linalg.solve(A_total, b_total)


def reprojection_errors(point_3d: np.ndarray, cameras: list[Camera], detections: list[tuple[float, float]]) -> np.ndarray:
    """Ошибка перепроецирования по каждой камере, пиксели. inf — точка позади камеры."""
    errors = []
    for camera, (u, v) in zip(cameras, detections):
        projected = camera.project(point_3d)
        if np.any(np.isnan(projected)):
            errors.append(np.inf)
        else:
            errors.append(float(np.hypot(projected[0] - u, projected[1] - v)))
    return np.array(errors)


def calculate_ray_error(point_3d: np.ndarray, cameras: list[Camera], detections: list[tuple[float, float]],) -> float:
    """Среднее расстояние от 3D-точки до лучей зрения камер (в единицах мира)."""
    errors = []

    for camera, (u, v) in zip(cameras, detections):

        direction = pixel_to_ray(camera, u, v)

        distance = np.linalg.norm(
            np.cross(
                point_3d - camera.position,
                direction,
            )
        )

        errors.append(distance)

    return float(np.mean(errors))


@dataclass
class TriangulationResult:
    point: np.ndarray                 # (3,) мировые координаты; NaN, если точку получить не удалось
    reprojection_errors: np.ndarray   # (N,) пиксели, для всех переданных камер по итоговой точке
    inliers: np.ndarray               # (N,) bool, какие камеры использованы в итоговой точке
    ok: bool                          # точка найдена и все её камеры прошли порог
    reason: str = ""                  # почему ok = False

    @property
    def num_inliers(self) -> int:
        return int(self.inliers.sum())


def triangulate_robust(
    cameras: list[Camera],
    detections: list[tuple[float, float]],
    max_reprojection_error_px: float = 20.0,
) -> TriangulationResult:
    """
    Триангуляция с отбрасыванием выбросов.

    1. Считаем точку по всем камерам.
    2. Пока есть камера с ошибкой перепроецирования выше порога (или точка позади неё)
       и камер больше двух, перебираем подмножества без одной камеры и оставляем то,
       у которого максимальная ошибка меньше. Это устойчивее, чем выбрасывать камеру
       с наибольшей ошибкой: выброс тянет точку на себя и портит ошибки других камер.
    3. При двух камерах выброс определить нельзя: если порог не пройден,
       точка возвращается с ok = False.
    """
    if len(cameras) != len(detections):
        raise ValueError("Количество камер и детекций не совпадает")

    n = len(cameras)
    all_idx = list(range(n))

    def solve(idx):
        try:
            point = triangulate([cameras[i] for i in idx], [detections[i] for i in idx])
        except np.linalg.LinAlgError:
            return None, None
        errors = reprojection_errors(point, [cameras[i] for i in idx], [detections[i] for i in idx])
        return point, errors

    def result(idx, point, ok, reason=""):
        inliers = np.zeros(n, dtype=bool)
        inliers[idx] = True
        errors = reprojection_errors(point, cameras, detections) if point is not None else np.full(n, np.nan)
        if point is None:
            point = np.full(3, np.nan)
        return TriangulationResult(point, errors, inliers, ok, reason)

    if n < 2:
        return result([], None, False, "меньше двух камер")

    idx = all_idx
    point, errors = solve(idx)
    if point is None:
        return result(idx, None, False, "лучи почти параллельны")

    while errors.max() > max_reprojection_error_px and len(idx) > 2:
        best = None
        for drop in idx:
            subset = [i for i in idx if i != drop]
            p, e = solve(subset)
            if p is not None and (best is None or e.max() < best[2].max()):
                best = (subset, p, e)
        if best is None:
            break
        idx, point, errors = best

    if errors.max() > max_reprojection_error_px:
        return result(idx, point, False, "ошибка перепроецирования выше порога")
    return result(idx, point, True)
