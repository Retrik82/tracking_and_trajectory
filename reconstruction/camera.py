import csv
import io
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger("peleng")

# CSV из Unreal Engine сохранены в Windows-кодировке
CSV_ENCODING = "cp1251"

# Физический размер матрицы (сенсора) камеры, мм. Постоянный и одинаковый для всех видеопотоков:
# от разрешения видео не зависит — с разрешением меняется только число пикселей на матрице.
SENSOR_WIDTH_MM = 23.76
SENSOR_HEIGHT_MM = 13.365

# Формулы f_x = f·W/23.76, f_y = f·H/13.365 верны, только если кадр занимает всю матрицу без обрезки.
# Тогда соотношение сторон кадра равно соотношению сторон матрицы (16:9); заметно другое соотношение
# (например, 1920×1088 с кодеком, дописавшим строки, или кадр 4:3) — обрезка/добавление полей, и
# внутренние параметры по этим формулам были бы неверны. Допуск — на округление размеров до целых.
ASPECT_TOLERANCE = 0.005

# Колонки CSV камеры: позиция в единицах мира Unreal (см), углы Rotator в градусах
POSE_COLUMNS = ("frame", "loc_x", "loc_y", "loc_z", "rot_roll", "rot_pitch", "rot_yaw")
# Фокус, мм: колонка Focal_Length_mm по кадрам (сцена 2) или пара «Focal length;35» в заголовке (сцена 1)
FOCAL_NAMES = ("focal_length_mm", "focal_length", "focallength")


class CameraCsvError(ValueError):
    """CSV камеры нельзя прочитать или интерпретировать (нет колонок, нет фокуса, нет ни одной позы)."""


class FrameNotFoundError(ValueError):
    """В CSV камеры нет корректной строки для запрошенного кадра (поза на этом кадре недоступна)."""


def _to_float(value: str) -> float:
    return float(value.strip().replace(",", "."))


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _is_number(value: str) -> bool:
    try:
        return math.isfinite(_to_float(value))
    except ValueError:
        return False


@dataclass(frozen=True)
class CameraCsv:
    """
    Содержимое CSV одной камеры.

    poses    {кадр: ((x, y, z), roll, pitch, yaw)} — позиция в единицах мира, углы в радианах;
    focal_mm {кадр: физическое фокусное расстояние, мм} — для тех же кадров, что poses;
    focal_source  "column" (значение в каждой строке) или "header" (одно значение в заголовке);
    issues   описание пропущенных строк (нечисловые / неполные / повторные / неположительный фокус).
    """

    path: str
    poses: dict
    focal_mm: dict
    focal_source: str
    issues: tuple

    def focal_length_mm(self, frame: int) -> float:
        if frame not in self.focal_mm:
            raise FrameNotFoundError(f"{self.path}: нет фокусного расстояния для кадра {frame}")
        return self.focal_mm[frame]


def _read_text(path: str) -> str:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):          # файл пересохранён в UTF-8 с BOM
        return raw[3:].decode("utf-8")
    return raw.decode(CSV_ENCODING)


def _focal_spec(path: str, header: list[str]) -> tuple[str, int, float | None]:
    """Где записан фокус: ("header", индекс, значение) или ("column", индекс колонки, None)."""
    found = []
    for i, cell in enumerate(header):
        if _normalize(cell) in FOCAL_NAMES:
            following = header[i + 1].strip() if i + 1 < len(header) else ""
            if following and _is_number(following):
                found.append(("header", i, _to_float(following)))
            else:
                found.append(("column", i, None))
    if not found:
        raise CameraCsvError(
            f"{path}: нет фокусного расстояния — ожидается колонка Focal_Length_mm или пара "
            f"«Focal length;<мм>» в заголовке; заголовок: {';'.join(header)}")
    if len(found) > 1:
        raise CameraCsvError(f"{path}: фокусное расстояние указано несколько раз ({len(found)}) — "
                             f"неясно, какое использовать; заголовок: {';'.join(header)}")
    return found[0]


@lru_cache(maxsize=None)
def read_camera_csv(file_path: str) -> CameraCsv:
    """
    Читает CSV камеры целиком (кодировка CP1251, разделитель «;», десятичная точка или запятая).

    Колонки находятся по именам заголовка (Frame, Loc_X, Loc_Y, Loc_Z, Rot_Roll, Rot_Pitch, Rot_Yaw;
    регистр и пробелы не важны), а не по позициям. Фокус — в мм (см. FOCAL_NAMES). Строки с
    нечисловыми, пустыми или неполными ячейками и с неположительным фокусом пропускаются — для таких
    кадров поза недоступна — и перечисляются в issues (значения не угадываются). Файл кешируется,
    чтобы при проходе по тысячам кадров не перечитывать его каждый раз.
    """
    path = str(file_path)
    try:
        text = _read_text(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"нет CSV камеры {path}") from None
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=";")
    header = next(reader, None)
    if not header:
        raise CameraCsvError(f"{path}: пустой файл")
    names = [_normalize(h) for h in header]
    missing = [c for c in POSE_COLUMNS if c not in names]
    if missing:
        raise CameraCsvError(
            f"{path}: в заголовке нет колонок {', '.join(missing)}; ожидается "
            f"Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw и фокус; заголовок: {';'.join(header)}")
    col = {c: names.index(c) for c in POSE_COLUMNS}
    source, focal_col, header_focal = _focal_spec(path, header)
    if source == "header" and not header_focal > 0:
        raise CameraCsvError(f"{path}: фокусное расстояние в заголовке {header_focal:g} мм — должно быть > 0")

    value_cols = [col[c] for c in POSE_COLUMNS[1:]] + ([focal_col] if source == "column" else [])
    poses, focal, issues, seen = {}, {}, [], set()
    for line, row in enumerate(reader, start=2):
        if not row or not any(cell.strip() for cell in row):
            continue
        cell = row[col["frame"]].strip() if len(row) > col["frame"] else ""
        try:
            frame = int(cell)
        except ValueError:
            issues.append(f"строка {line}: номер кадра {cell!r}")
            continue
        if len(row) <= max(value_cols):
            issues.append(f"кадр {frame}: неполная строка ({len(row)} ячеек)")
            continue
        bad = [header[i] for i in value_cols if not _is_number(row[i])]
        if bad:
            issues.append(f"кадр {frame}: нечисловые или пустые значения в {', '.join(bad)}")
            continue
        values = [_to_float(row[i]) for i in value_cols[:6]]
        f_mm = _to_float(row[focal_col]) if source == "column" else header_focal
        if not f_mm > 0:
            issues.append(f"кадр {frame}: фокусное расстояние {f_mm:g} мм — должно быть > 0")
            continue
        if frame in seen:
            # две разные позы одного кадра — какая верна, неизвестно: кадр исключается целиком
            issues.append(f"кадр {frame}: повторяется")
            poses.pop(frame, None)
            focal.pop(frame, None)
            continue
        seen.add(frame)
        x, y, z, roll, pitch, yaw = values
        poses[frame] = ((x, y, z), math.radians(roll), math.radians(pitch), math.radians(yaw))
        focal[frame] = f_mm
    if not poses:
        raise CameraCsvError(f"{path}: нет ни одной корректной строки позы"
                             + (f" ({'; '.join(issues[:3])})" if issues else ""))
    if issues:
        log.warning("%s: пропущено строк с некорректными значениями: %d (%s%s) — для этих кадров поза "
                    "камеры недоступна", path, len(issues), "; ".join(issues[:3]), "; …" if len(issues) > 3 else "")
    return CameraCsv(path, poses, focal, source, tuple(issues))


def load_camera_poses(file_path: str) -> dict[int, tuple[tuple[float, float, float], float, float, float]]:
    """
    Читает CSV камеры целиком: {кадр: ((x, y, z), roll, pitch, yaw)}, углы в радианах.

    Обёртка над read_camera_csv (там же фокус по кадрам и список пропущенных строк).
    Строки с нечисловыми ячейками пропускаются (для таких кадров поза недоступна).
    """
    return read_camera_csv(str(file_path)).poses


def check_image_size(width, height) -> tuple[int, int]:
    """Ширина и высота кадра в пикселях: положительные целые, иначе ValueError."""
    out = []
    for name, value in (("ширина", width), ("высота", height)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
                or not math.isfinite(value) or int(value) != value or value <= 0:
            raise ValueError(f"{name} кадра {value!r} — ожидается положительное целое число пикселей")
        out.append(int(value))
    return out[0], out[1]


def pixel_focal_lengths(focal_length_mm: float, width: int, height: int) -> tuple[float, float]:
    """
    Физический фокус (мм) -> фокусные расстояния в пикселях для кадра width × height:

        f_x = f_мм · W_px / 23.76,   f_y = f_мм · H_px / 13.365

    (матрица постоянная, SENSOR_WIDTH_MM × SENSOR_HEIGHT_MM; кадр занимает её целиком). Единственное
    место пересчёта мм -> пиксели в проекте. Кадр с другим соотношением сторон, чем у матрицы, —
    обрезка или поля: формулы к нему неприменимы, ValueError (произвольные параметры не подставляются).
    """
    if not (isinstance(focal_length_mm, (int, float, np.integer, np.floating))
            and math.isfinite(focal_length_mm) and focal_length_mm > 0):
        raise ValueError(f"фокусное расстояние {focal_length_mm!r} мм — ожидается положительное число")
    width, height = check_image_size(width, height)
    aspect, sensor_aspect = width / height, SENSOR_WIDTH_MM / SENSOR_HEIGHT_MM
    if abs(aspect / sensor_aspect - 1.0) > ASPECT_TOLERANCE:
        raise ValueError(
            f"кадр {width}×{height} (соотношение сторон {aspect:.4f}) не соответствует матрице "
            f"{SENSOR_WIDTH_MM} × {SENSOR_HEIGHT_MM} мм ({sensor_aspect:.4f}): изображение обрезано или "
            f"дополнено, и пересчёт фокуса по всей ширине/высоте матрицы был бы неверен")
    return float(focal_length_mm) * width / SENSOR_WIDTH_MM, float(focal_length_mm) * height / SENSOR_HEIGHT_MM


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

        # Размер кадра в пикселях — того видеопотока, к которому относится камера
        self.width, self.height = check_image_size(width, height)
        for name, value in (("f_x", f_x), ("f_y", f_y)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} = {value!r} px — ожидается положительное число")
        # Физическое фокусное расстояние, мм (None — камера задана сразу пиксельными параметрами).
        # От разрешения не зависит; f_x, f_y — его пиксельное представление для width × height.
        self.focal_length_mm: float | None = None
        self._set_intrinsics(f_x, f_y, c_x, c_y)

        # Строим матрицу поворота R на основе переданных углов
        self.update_rotation(roll, pitch, yaw)

    def _set_intrinsics(self, f_x: float, f_y: float, c_x: float, c_y: float) -> None:
        self.f_x = float(f_x)
        self.f_y = float(f_y)
        self.c_x = float(c_x)
        self.c_y = float(c_y)
        # Матрица внутренних параметров камеры K
        self.K = np.array([
            [self.f_x, 0, self.c_x],
            [0, self.f_y, self.c_y],
            [0,  0,    1] ], dtype=float)

    @classmethod
    def from_physical(cls, position, roll: float, pitch: float, yaw: float,
                      focal_length_mm: float, width: int, height: int) -> "Camera":
        """
        Камера по физическим параметрам: фокус в мм, матрица SENSOR_WIDTH_MM × SENSOR_HEIGHT_MM (постоянная)
        и фактическое разрешение кадра width × height этого видеопотока. Пиксельные f_x, f_y — из
        pixel_focal_lengths (один пересчёт), главная точка — центр кадра (c_x = W/2, c_y = H/2, как и прежде).
        """
        width, height = check_image_size(width, height)
        f_x, f_y = pixel_focal_lengths(focal_length_mm, width, height)
        camera = cls(position, roll, pitch, yaw, f_x, f_y, width / 2, height / 2, width, height)
        camera.focal_length_mm = float(focal_length_mm)
        return camera

    @property
    def sensor_size_mm(self) -> tuple[float, float]:
        """Физический размер матрицы, мм — постоянный, от разрешения не зависит."""
        return SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM

    def set_focal_length_mm(self, focal_length_mm: float) -> None:
        """Новый физический фокус (зум): пиксельные f_x, f_y и K пересчитываются для текущего кадра."""
        f_x, f_y = pixel_focal_lengths(focal_length_mm, self.width, self.height)
        self._set_intrinsics(f_x, f_y, self.c_x, self.c_y)
        self.focal_length_mm = float(focal_length_mm)

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
            raise FrameNotFoundError(f"Кадр {frame_number} не найден в файле {file_path}")

        position, roll, pitch, yaw = poses[frame_number]
        return cls(
            position=position,
            roll=roll, pitch=pitch, yaw=yaw,
            f_x=f_x, f_y=f_y,
            c_x=c_x, c_y=c_y,
            width=width, height=height
        )

    @classmethod
    def from_csv(cls, file_path: str, frame_number: int, width: int, height: int) -> "Camera":
        """
        Камера целиком из CSV: позиция, углы и физический фокус (мм) из строки кадра frame_number;
        width × height — фактическое разрешение видео этой камеры (от него — пиксельные f_x, f_y).
        """
        data = read_camera_csv(str(file_path))
        if frame_number not in data.poses:
            raise FrameNotFoundError(f"Кадр {frame_number} не найден в файле {file_path}")
        position, roll, pitch, yaw = data.poses[frame_number]
        return cls.from_physical(position, roll, pitch, yaw, data.focal_length_mm(frame_number), width, height)

    def update_from_csv_frame(self, file_path: str, frame_number: int):
        """
        Метод для существующего объекта камеры: обновляет позицию и вращение из файла. Если камера
        задана физическим фокусом (from_physical / from_csv), обновляется и он — при зуме по кадрам
        пиксельные f_x, f_y пересчитываются для того же разрешения.
        """
        data = read_camera_csv(str(file_path))
        if frame_number not in data.poses:
            return False

        position, roll, pitch, yaw = data.poses[frame_number]
        self.position = np.asarray(position, dtype=float)
        self.update_rotation(roll, pitch, yaw)
        focal = data.focal_mm.get(frame_number)
        if self.focal_length_mm is not None and focal is not None and focal != self.focal_length_mm:
            self.set_focal_length_mm(focal)
        return True
