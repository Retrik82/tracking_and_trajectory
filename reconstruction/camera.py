import csv
import math
from functools import lru_cache

import numpy as np

# CSV из Unreal Engine сохранены в Windows-кодировке
CSV_ENCODING = "cp1251"


def _to_float(value: str) -> float:
    return float(value.strip().replace(",", "."))


@lru_cache(maxsize=None)
def load_camera_poses(file_path: str) -> dict[int, tuple[tuple[float, float, float], float, float, float]]:
    """
    Читает CSV камеры целиком: {кадр: ((x, y, z), roll, pitch, yaw)}, углы в радианах.

    Файл кешируется, чтобы при проходе по 1000 кадрам не перечитывать его каждый раз.
    Строки с нечисловыми ячейками пропускаются (для таких кадров поза недоступна).
    """
    poses = {}
    with open(file_path, mode="r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # Пропускаем строку заголовка (в ней же указан фокус)

        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                frame = int(row[0])
                position = (_to_float(row[1]), _to_float(row[2]), _to_float(row[3]))
                roll, pitch, yaw = (math.radians(_to_float(v)) for v in row[4:7])
            except (ValueError, IndexError):
                continue
            poses[frame] = (position, roll, pitch, yaw)
    return poses


def camera_axes(roll: float, pitch: float, yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Оси камеры в мировой системе Unreal Engine (левая, Z вверх). Углы в радианах.

    Возвращает (forward, right, up) — формулы из модели камеры проекта.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    forward = np.array([cp * cy, cp * sy, sp])
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return forward, right, up


class Camera:
    def __init__(
        self,
        position: tuple[float, float, float],
        roll: float,   # Передаем углы в радианах
        pitch: float,
        yaw: float,
        f_x: float,
        f_y: float,
        c_x: float,
        c_y: float,
        width: int,
        height: int,
    ):
        self.position = np.asarray(position, dtype=float)

        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y
        self.width = width
        self.height = height

        # Матрица внутренних параметров камеры K
        self.K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0,  0,    1] ], dtype=float)

        # Строим матрицу поворота R на основе переданных углов
        self.update_rotation(roll, pitch, yaw)

    def update_rotation(self, roll: float, pitch: float, yaw: float):
        """
        Строит матрицу поворота мир -> камера.

        Строки R — оси камеры в мире: x = right, y = -up (ось v пикселей направлена вниз),
        z = forward. Тогда для d = P - C: (x, y, z) = R @ d и
        u = c_x + f_x * x / z, v = c_y + f_y * y / z, что совпадает с
        u = c_x + f_x * (d·right) / z, v = c_y - f_y * (d·up) / z.
        """
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.forward, self.right, self.up = camera_axes(roll, pitch, yaw)
        self.R = np.vstack([self.right, -self.up, self.forward])

    def project(self, points: np.ndarray) -> np.ndarray:
        """
        Проецирует мировые точки (3,) или (N, 3) в пиксели (2,) или (N, 2).

        Для точек позади камеры (z <= 0) возвращает NaN.
        """
        points = np.asarray(points, dtype=float)
        cam = (points - self.position) @ self.R.T
        z = cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.c_x + self.f_x * cam[..., 0] / z
            v = self.c_y + self.f_y * cam[..., 1] / z
        uv = np.stack([u, v], axis=-1)
        uv[z <= 0] = np.nan
        return uv

    def depth(self, points: np.ndarray) -> np.ndarray:
        """Глубина точек вдоль оси forward (в единицах мира)."""
        return (np.asarray(points, dtype=float) - self.position) @ self.forward

    @classmethod
    def from_csv_frame(
        cls,
        file_path: str,
        frame_number: int,
        f_x: float,
        f_y: float,
        c_x: float,
        c_y: float,
        width: int,
        height: int
    ):
        """
        Фабричный метод: берёт позу камеры из строки CSV с номером frame_number
        и создает объект Camera с вашими f_x и f_y.
        """
        poses = load_camera_poses(str(file_path))
        if frame_number not in poses:
            raise ValueError(f"Кадр {frame_number} не найден в файле {file_path}")

        position, roll, pitch, yaw = poses[frame_number]
        return cls(
            position=position,
            roll=roll, pitch=pitch, yaw=yaw,
            f_x=f_x, f_y=f_y,
            c_x=c_x, c_y=c_y,
            width=width, height=height
        )

    def update_from_csv_frame(self, file_path: str, frame_number: int):
        """Метод для существующего объекта камеры: обновляет только позицию и вращение из файла."""
        poses = load_camera_poses(str(file_path))
        if frame_number not in poses:
            return False

        position, roll, pitch, yaw = poses[frame_number]
        self.position = np.asarray(position, dtype=float)
        self.update_rotation(roll, pitch, yaw)
        return True
