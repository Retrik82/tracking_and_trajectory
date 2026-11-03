import os
from pathlib import Path

import pytest

from reconstruction.camera import Camera

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PELENG_DATA_DIR", REPO_ROOT / "tmp" / "project2"))

WIDTH, HEIGHT = 1920, 1080
SENSOR_W_MM, SENSOR_H_MM = 23.76, 13.365
FOCAL_MM = {1: 35.0, 2: 60.0, 3: 35.0, 4: 50.0, 5: 20.0}


def camera_csv(cam_id: int) -> Path:
    return DATA_DIR / f"object_transform_CineCameraActor{cam_id}.csv"


def make_camera(cam_id: int, frame: int) -> Camera:
    f = FOCAL_MM[cam_id]
    return Camera.from_csv_frame(
        camera_csv(cam_id), frame,
        f_x=f / SENSOR_W_MM * WIDTH, f_y=f / SENSOR_H_MM * HEIGHT,
        c_x=WIDTH / 2, c_y=HEIGHT / 2, width=WIDTH, height=HEIGHT,
    )


requires_data = pytest.mark.skipif(
    not camera_csv(1).exists(),
    reason=f"нет данных проекта в {DATA_DIR} (задайте PELENG_DATA_DIR)",
)


def load_target_positions() -> dict[int, tuple[float, float, float]]:
    """Положения цели по кадрам; строки с испорченными ячейками (кадры 402, 691) пропускаются."""
    import csv

    positions = {}
    with open(DATA_DIR / "object_transform_SM_AA_track.csv", encoding="cp1251", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            try:
                positions[int(row[0])] = tuple(float(v) for v in row[1:4])
            except (ValueError, IndexError):
                continue
    return positions
