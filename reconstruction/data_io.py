"""Чтение CSV проекта: позы камер и положение цели (только для оценки качества)."""

import csv
import re
from dataclasses import dataclass, field
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
    return Path(data_dir) / f"Way1.FinalImage.CineCameraActor{cam_id}.mp4"


def target_csv_path(data_dir: Path) -> Path:
    return Path(data_dir) / "object_transform_SM_AA_track.csv"


def read_focal_length_mm(csv_path: Path) -> float:
    """Фокус записан в строке заголовка: «...;;;Focal length;35»."""
    with open(csv_path, encoding=CSV_ENCODING, newline="") as f:
        header = next(csv.reader(f, delimiter=";"))
    for name, value in zip(header, header[1:]):
        if name.strip().lower() == "focal length":
            return float(value.replace(",", "."))
    raise ValueError(f"В заголовке {csv_path} нет фокусного расстояния")


def make_camera(data_dir: Path, cam_id: int, csv_frame: int) -> Camera:
    """Камера с позой из строки CSV с номером csv_frame и фокусом из заголовка."""
    path = camera_csv_path(data_dir, cam_id)
    focal = read_focal_length_mm(path)
    return Camera.from_csv_frame(
        path, csv_frame,
        f_x=focal / SENSOR_WIDTH_MM * IMAGE_WIDTH,
        f_y=focal / SENSOR_HEIGHT_MM * IMAGE_HEIGHT,
        c_x=IMAGE_WIDTH / 2, c_y=IMAGE_HEIGHT / 2,
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
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
