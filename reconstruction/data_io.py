"""Чтение CSV проекта: позы камер и положение цели (только для оценки качества)."""

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from reconstruction.camera import CSV_ENCODING, Camera

SENSOR_WIDTH_MM = 23.76
SENSOR_HEIGHT_MM = 13.365
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


def camera_csv_path(data_dir: Path, cam_id: int) -> Path:
    return Path(data_dir) / f"object_transform_CineCameraActor{cam_id}.csv"


def video_path(data_dir: Path, cam_id: int) -> Path:
    """Видео камеры: любой префикс сцены (Way1, Way2, …); нет файла — имя сцены 1 (для сообщений)."""
    matches = sorted(Path(data_dir).glob(f"*.FinalImage.CineCameraActor{cam_id}.mp4"))
    if matches:
        return matches[0]
    return Path(data_dir) / f"Way1.FinalImage.CineCameraActor{cam_id}.mp4"


def video_resolution(data_dir: Path, cam_id: int) -> tuple[int, int]:
    """
    Разрешение видео камеры (ширина, высота) из свойств самого файла — оно и используется
    в расчёте фокусных расстояний в пикселях. Видео нет (режим detections) — значения по
    умолчанию IMAGE_WIDTH × IMAGE_HEIGHT.
    """
    return _video_resolution(str(video_path(data_dir, cam_id)))


@lru_cache(maxsize=None)
def _video_resolution(path: str) -> tuple[int, int]:
    if Path(path).is_file():
        import cv2

        cap = cv2.VideoCapture(path)
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if width > 0 and height > 0:
            return width, height
    return IMAGE_WIDTH, IMAGE_HEIGHT


def target_csv_path(data_dir: Path) -> Path:
    return Path(data_dir) / "object_transform_SM_AA_track.csv"


def read_focal_length_mm(csv_path: Path) -> float:
    """
    Фокус камеры, мм. Два формата CSV:
      сцена 1: пара в заголовке «...;;;Focal length;35»;
      сцена 2: колонка Focal_Length_mm — значение из первой строки данных
               (по кадрам он постоянен; если бы менялся, нужна поддержка в Camera).
    """
    return _read_focal_length_mm(str(csv_path))


@lru_cache(maxsize=None)
def _read_focal_length_mm(csv_path: str) -> float:
    with open(csv_path, encoding=CSV_ENCODING, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        for name, value in zip(header, header[1:]):
            if name.strip().lower() == "focal length":
                return float(value.replace(",", "."))
        normalized = [h.strip().lower().replace(" ", "_") for h in header]
        for col, name in enumerate(normalized):
            if name.startswith("focal_length"):
                for row in reader:
                    if row and len(row) > col and row[col].strip():
                        return float(row[col].strip().replace(",", "."))
    raise ValueError(f"В {csv_path} нет фокусного расстояния (ни в заголовке, ни колонкой)")


def make_camera(data_dir: Path, cam_id: int, csv_frame: int) -> Camera:
    """
    Камера с позой из строки CSV с номером csv_frame и фокусом из CSV. Разрешение берётся
    из свойств видеофайла этой камеры (фокусные расстояния в пикселях считаются от него);
    без видео — значения по умолчанию 1920 × 1080.
    """
    path = camera_csv_path(data_dir, cam_id)
    focal = read_focal_length_mm(path)
    width, height = video_resolution(data_dir, cam_id)
    return Camera.from_csv_frame(
        path, csv_frame,
        f_x=focal / SENSOR_WIDTH_MM * width,
        f_y=focal / SENSOR_HEIGHT_MM * height,
        c_x=width / 2, c_y=height / 2,
        width=width, height=height,
    )


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
