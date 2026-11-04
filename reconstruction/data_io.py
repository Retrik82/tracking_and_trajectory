"""
Чтение данных проекта: позы и фокус камер (CSV), разрешение видео, положение цели (только для оценки).

Путь параметров камеры:
  CSV камеры N (object_transform_CineCameraActor{N}.csv) -> позиция, углы, фокус в мм (camera.read_camera_csv);
  видео камеры N (*.FinalImage.CineCameraActor{N}.mp4)   -> ширина и высота кадра (probe_video_resolution);
  make_camera(..., image_size)                             -> Camera.from_physical: f_x, f_y в пикселях
                                                              от постоянной матрицы 23.76 × 13.365 мм.
Без видео (готовые детекции) разрешение берётся из meta.json детекций (image_size по камерам),
куда его записывает шаг detect по тем же видео. Заданного заранее «общего» разрешения нет.
"""

import csv
import json
import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from reconstruction.camera import (  # noqa: F401 — SENSOR_* реэкспорт для прежних импортов
    CSV_ENCODING,
    SENSOR_HEIGHT_MM,
    SENSOR_WIDTH_MM,
    Camera,
    CameraCsvError,
    check_image_size,
    read_camera_csv,
)

log = logging.getLogger("peleng")

# Видео камеры N любой сцены (Way1, Way2, …); номер камеры — ровно N (Actor1 не совпадает с Actor10)
VIDEO_PATTERN = "*.FinalImage.CineCameraActor{cam}.mp4"
DETECTIONS_META = "meta.json"


class VideoResolutionError(RuntimeError):
    """Разрешение видео не удалось определить (нет файла, не открывается, не декодируется, нулевой размер)."""


def camera_csv_path(data_dir: Path, cam_id: int) -> Path:
    return Path(data_dir) / f"object_transform_CineCameraActor{cam_id}.csv"


def find_video(data_dir: Path, cam_id: int, pattern: str = VIDEO_PATTERN) -> Path | None:
    """
    Видео камеры cam_id по шаблону (по умолчанию — любой префикс сцены). None — файла нет.
    Несколько подходящих файлов (например, Way1 и Way2 в одной папке) — ошибка: неясно, какое видео
    относится к камере, а от него зависят разрешение и внутренние параметры.
    """
    matches = sorted(p for p in Path(data_dir).glob(pattern.format(cam=cam_id)) if p.is_file())
    if len(matches) > 1:
        raise VideoResolutionError(
            f"cam{cam_id}: несколько видео в {data_dir} ({', '.join(p.name for p in matches)}) — неясно, какое "
            f"относится к камере; оставьте одно или задайте точный шаблон имени")
    return matches[0] if matches else None


def video_path(data_dir: Path, cam_id: int) -> Path:
    """Видео камеры; нет файла — путь с шаблоном имени (не существует; для .exists() и сообщений)."""
    return find_video(data_dir, cam_id) or Path(data_dir) / VIDEO_PATTERN.format(cam=cam_id)


def target_csv_path(data_dir: Path) -> Path:
    return Path(data_dir) / "object_transform_SM_AA_track.csv"


def read_focal_length_mm(csv_path: Path, frame: int | None = None) -> float:
    """
    Физический фокус камеры, мм, из CSV (пара «Focal length;35» в заголовке или колонка Focal_Length_mm).
    frame=None — фокус файла; если он меняется по кадрам (зум), нужен номер кадра — иначе ошибка.
    """
    data = read_camera_csv(str(csv_path))
    if frame is not None:
        return data.focal_length_mm(frame)
    values = set(data.focal_mm.values())
    if len(values) > 1:
        raise CameraCsvError(f"{csv_path}: фокус меняется по кадрам ({min(values):g}–{max(values):g} мм) — "
                             f"укажите кадр")
    return values.pop()


# ---------------------------------------------------------------- разрешение видео

def probe_video_resolution(path: Path | str) -> tuple[int, int]:
    """
    Фактическое разрешение видеофайла (ширина, высота), px. Берётся из метаданных (CAP_PROP_FRAME_WIDTH /
    CAP_PROP_FRAME_HEIGHT) и проверяется по первому декодированному кадру: при расхождении верен
    декодированный кадр — именно в его пикселях детектор выдаёт рамки. Кешируется по файлу (путь,
    время изменения, размер). Ошибка — VideoResolutionError; произвольное разрешение не подставляется.
    """
    p = Path(path)
    if not p.is_file():
        raise VideoResolutionError(f"нет видео {p} — разрешение не определить")
    st = p.stat()
    return _probe_file(str(p.resolve()), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=None)
def _probe_file(path: str, _mtime_ns: int, _size: int) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoResolutionError(f"видео {path} не открывается (файл повреждён или нет декодера)")
        meta_w, meta_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        ok, image = cap.read()
    finally:
        cap.release()
    return frame_resolution(image if ok else None, path, (meta_w, meta_h))


def frame_resolution(image, source: str, metadata: tuple[float, float] | None = None) -> tuple[int, int]:
    """Разрешение по декодированному кадру (с проверкой метаданных источника, если они есть)."""
    meta_ok = metadata is not None and all(math.isfinite(v) and v > 0 for v in metadata)
    meta_text = f"{metadata[0]:g}×{metadata[1]:g}" if metadata is not None else "нет"
    if image is None or getattr(image, "ndim", 0) < 2 or image.shape[0] == 0 or image.shape[1] == 0:
        raise VideoResolutionError(f"{source}: не удалось декодировать кадр — разрешение не определить "
                                   f"(метаданные: {meta_text})")
    height, width = int(image.shape[0]), int(image.shape[1])
    if metadata is not None and not meta_ok:
        log.warning("%s: в метаданных нет разрешения (%s) — взято по декодированному кадру %d×%d",
                    source, meta_text, width, height)
    elif meta_ok and (round(metadata[0]), round(metadata[1])) != (width, height):
        log.warning("%s: метаданные контейнера %s, а декодированный кадр %d×%d — используется размер кадра",
                    source, meta_text, width, height)
    return width, height


def _read_meta(path: Path) -> dict | None:
    """meta.json детекций; нет файла — None. Старые файлы могли быть записаны в кодировке системы."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode(CSV_ENCODING)
        meta = json.loads(text)
    except ValueError as e:
        raise VideoResolutionError(f"{path}: не читается ({e})") from None
    if not isinstance(meta, dict):
        raise VideoResolutionError(f"{path}: ожидается JSON-объект")
    return meta


def detections_image_sizes(det_dir: Path) -> dict[int, tuple[int, int]]:
    """Разрешение видео по камерам из meta.json детекций ({камера: (ширина, высота)}); нет — пусто."""
    path = Path(det_dir) / DETECTIONS_META
    meta = _read_meta(path) or {}
    out = {}
    for cam, size in (meta.get("image_size") or {}).items():
        try:
            out[int(cam)] = check_image_size(*size)
        except (TypeError, ValueError) as e:
            raise VideoResolutionError(f"{path}: некорректное image_size камеры {cam}: {size!r} ({e})") from None
    return out


def write_detections_image_sizes(det_dir: Path, sizes: dict[int, tuple[int, int]]) -> None:
    """Дописать разрешение видео по камерам в meta.json детекций (остальные поля не трогаются)."""
    path = Path(det_dir) / DETECTIONS_META
    meta = _read_meta(path) or {}            # нечитаемый файл — ошибка, а не перезапись без прежних полей
    merged = {str(k): list(v) for k, v in (meta.get("image_size") or {}).items()}
    merged.update({str(cam): [int(w), int(h)] for cam, (w, h) in sizes.items()})
    meta["image_size"] = dict(sorted(merged.items(), key=lambda kv: int(kv[0])))
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_image_sizes(data_dir: Path, cameras, det_dir: Path | None = None,
                        pattern: str = VIDEO_PATTERN) -> dict[int, tuple[int, int]]:
    """
    Разрешение кадра каждой камеры, отдельно: из её видео; видео нет — из meta.json детекций.
    Видео и детекции расходятся по размеру — ошибка (детекции сделаны на другом видео, их пиксели
    не согласуются с внутренними параметрами). Ни того ни другого — ошибка с указанием, чего не хватает.
    """
    from_meta = detections_image_sizes(det_dir) if det_dir is not None else {}
    out = {}
    for cam in cameras:
        cam = int(cam)
        video = find_video(data_dir, cam, pattern)
        if video is not None:
            size = probe_video_resolution(video)
            if cam in from_meta and from_meta[cam] != size:
                raise VideoResolutionError(
                    f"cam{cam}: видео {video.name} {size[0]}×{size[1]}, а детекции в {det_dir} сделаны на кадрах "
                    f"{from_meta[cam][0]}×{from_meta[cam][1]} — пересчитайте детекции (--redetect)")
            out[cam] = size
        elif cam in from_meta:
            out[cam] = from_meta[cam]
        else:
            where = f" и нет image_size в {Path(det_dir) / DETECTIONS_META}" if det_dir is not None else ""
            raise VideoResolutionError(
                f"cam{cam}: разрешение кадра неизвестно — нет видео {pattern.format(cam=cam)} в {data_dir}{where}")
    return out


# ---------------------------------------------------------------- камеры

def make_camera(data_dir: Path, cam_id: int, csv_frame: int, image_size: tuple[int, int] | None = None) -> Camera:
    """
    Камера cam_id на кадре CSV csv_frame: позиция, углы и физический фокус (мм) — из её CSV;
    image_size (ширина, высота) — фактическое разрешение её видео; None — определить по видеофайлу
    камеры. Пиксельные f_x, f_y и K — Camera.from_physical (матрица 23.76 × 13.365 мм).
    """
    if image_size is None:
        video = find_video(data_dir, cam_id)
        if video is None:
            raise VideoResolutionError(f"cam{cam_id}: нет видео {VIDEO_PATTERN.format(cam=cam_id)} в {data_dir} — "
                                       f"разрешение кадра для внутренних параметров не определить")
        image_size = probe_video_resolution(video)
    width, height = check_image_size(*image_size)
    return Camera.from_csv(camera_csv_path(data_dir, cam_id), csv_frame, width, height)


@dataclass
class TargetTrack:
    frames: np.ndarray                 # (N,) номера кадров CSV
    positions: np.ndarray              # (N, 3) в единицах мира; NaN в испорченных ячейках
    issues: list[str] = field(default_factory=list)   # описание испорченных ячеек

    def position(self, frame: int) -> np.ndarray:
        """Положение цели на кадре CSV; NaN, если кадра нет или ячейка испорчена."""
        idx = np.searchsorted(self.frames, frame)
        if idx < len(self.frames) and self.frames[idx] == frame:
            return self.positions[idx]
        return np.full(3, np.nan)

    @property
    def valid(self) -> np.ndarray:
        return np.all(np.isfinite(self.positions), axis=1)


_NUMBER = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?$")


def load_target(data_dir: Path) -> TargetTrack:
    """
    Читает CSV цели. Нечисловые ячейки (например, «авг.13» — число, превращённое
    Excel в дату) не угадываются, а заменяются на NaN и попадают в issues.
    """
    path = target_csv_path(data_dir)
    frames, positions, issues = [], [], []
    with open(path, encoding=CSV_ENCODING, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            frame = int(row[0])
            values = []
            for col in range(1, 4):
                cell = row[col].strip() if col < len(row) else ""
                if _NUMBER.match(cell):
                    values.append(float(cell.replace(",", ".")))
                else:
                    values.append(np.nan)
                    issues.append(f"кадр {frame}, {header[col]} = {cell!r}")
            frames.append(frame)
            positions.append(values)
    return TargetTrack(np.array(frames), np.array(positions, dtype=float), issues)
