"""
Синтетическая сцена в формате проекта: CSV камер, CSV цели и «детекции» по коробке грузовика.

Нужна, чтобы прогнать весь пайплайн (кроме detect и video) без видео, весов YOLO и 3D-модели:
проверить код, посмотреть отчёт и графики, сравнить методы на данных с известной правдой.
Это НЕ замена реальным данным: цифры на синтетике ничего не говорят о точности на видео.

Что моделируется (похоже на реальную сцену, см. README):
- грузовик — коробка DEFAULT_DIMENSIONS (16.5 × 2.55 × 4 м), едет по петле с переменной
  скоростью (до ~25 м/с, продольное ускорение до ~5 м/с², поперечное до ~6 м/с²);
- точка CSV цели — низ передней грани коробки (то, что трекер выдаёт как position);
- 5 камер с фокусами 35/60/35/50/20 мм; Camera5 поворачивается по yaw на части кадров;
- детекции: рамка проекции коробки с шумом, центр маски — проекция центра коробки с шумом,
  уверенность случайная; участки потери видимости: кадры 200–300 видит одна камера,
  кадры 500–514 не видит никто, Cam4 «заезжает за препятствие» (ребро съёживается) и пропадает,
  редкие выбросы (ложная рамка).

Запуск:
    python tools/make_synthetic_data.py --data tmp/synthetic --out outputs/synthetic
    python main.py --data tmp/synthetic --out outputs/synthetic --steps center,optimize,cab,realtime,approx,compare
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.schema import Detection                       # noqa: E402
from reconstruction.cab_model import DEFAULT_DIMENSIONS, box_corners  # noqa: E402
from reconstruction.camera import Camera                     # noqa: E402
from reconstruction.data_io import (                         # noqa: E402
    IMAGE_HEIGHT, IMAGE_WIDTH, SENSOR_HEIGHT_MM, SENSOR_WIDTH_MM, camera_csv_path, target_csv_path,
)

NUM_FRAMES = 1001            # кадров видео
CSV_ROWS = 1002              # строк в CSV (как в реальных данных)
FPS = 30.0
FOCAL_MM = {1: 35.0, 2: 60.0, 3: 35.0, 4: 50.0, 5: 20.0}


def trajectory(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Опорная точка (низ передней грани), см, и скорость, см/кадр. Петля с переменной скоростью."""
    t = frames / FPS
    speed = 1600.0 + 800.0 * np.sin(2 * np.pi * t / 15.0)             # см/с: 8–24 м/с
    # пройденный путь интегрированием скорости
    dt = 1.0 / FPS
    path = np.concatenate([[0.0], np.cumsum((speed[1:] + speed[:-1]) / 2 * dt)])
    a, b = 12000.0, 7000.0                                             # полуоси эллипса, см
    perimeter = 2 * np.pi * np.sqrt((a ** 2 + b ** 2) / 2)
    theta = 2 * np.pi * path / perimeter
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    z = 300.0 + 150.0 * np.sin(theta) + 40.0 * np.sin(3 * theta)      # лёгкий рельеф
    pos = np.c_[x, y, z]
    vel = np.gradient(pos, axis=0)
    return pos, vel


def look_at(position, target, focal_mm):
    d = np.asarray(target, float) - np.asarray(position, float)
    pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
    yaw = math.atan2(d[1], d[0])
    return Camera(position, 0.0, pitch, yaw,
                  focal_mm / SENSOR_WIDTH_MM * IMAGE_WIDTH, focal_mm / SENSOR_HEIGHT_MM * IMAGE_HEIGHT,
                  IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, IMAGE_WIDTH, IMAGE_HEIGHT)


def camera_poses(pos: np.ndarray) -> dict[int, list[Camera]]:
    """Позы камер по кадрам. Camera5 на кадрах 565–960 поворачивается за грузовиком."""
    center = pos.mean(axis=0)
    places = {1: (-22000.0, -14000.0, 2500.0), 2: (24000.0, -12000.0, 3200.0), 3: (20000.0, 15000.0, 1800.0),
              4: (-18000.0, 13000.0, 2200.0), 5: (0.0, -17000.0, 1500.0)}
    aims = {1: center + [0, 0, 0], 2: center + [3000, 0, 0], 3: center + [0, 2000, 0], 4: center + [-2000, 2000, 0]}
    out = {}
    for cam in (1, 2, 3, 4):
        out[cam] = [look_at(places[cam], aims[cam], FOCAL_MM[cam])] * CSV_ROWS
    cams5 = []
    base = look_at(places[5], center + [0, 3000, 0], FOCAL_MM[5])
    for f in range(CSV_ROWS):
        if 565 <= f <= 960:
            target = pos[min(f, NUM_FRAMES - 1)].copy()
            target[2] = center[2]
            cams5.append(look_at(places[5], target, FOCAL_MM[5]))
        else:
            cams5.append(base)
    out[5] = cams5
    return out


def write_camera_csv(path: Path, cams: list[Camera], focal_mm: float) -> None:
    with open(path, "w", encoding="cp1251", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\r\n")
        w.writerow(["Frame", "Loc_X", "Loc_Y", "Loc_Z", "Rot_Roll", "Rot_Pitch", "Rot_Yaw", "", "", "Focal length",
                    f"{focal_mm:g}"])
        for frame, c in enumerate(cams):
            w.writerow([frame, *(f"{v:.3f}" for v in c.position),
                        f"{math.degrees(c.roll):.4f}", f"{math.degrees(c.pitch):.4f}", f"{math.degrees(c.yaw):.4f}"])


def write_target_csv(path: Path, pos: np.ndarray) -> None:
    with open(path, "w", encoding="cp1251", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\r\n")
        w.writerow(["Frame", "Loc_X", "Loc_Y", "Loc_Z", "Rot_Roll", "Rot_Pitch", "Rot_Yaw"])
        for frame in range(CSV_ROWS):
            p = pos[min(frame, NUM_FRAMES - 1)]
            cells = [f"{v:.3f}" for v in p]
            if frame == 402:
                cells[0] = "авг.13"             # как в реальном CSV: Excel превратил число в дату
            if frame == 691:
                cells[1] = "июл.25"
            w.writerow([frame, *cells, "0", "0", "0"])


def visible_box(cam: Camera, corners: np.ndarray):
    uv = cam.project(corners)
    if not np.all(np.isfinite(uv)):
        return None
    x1, x2, y1, y2 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
    if x2 < 0 or x1 > IMAGE_WIDTH or y2 < 0 or y1 > IMAGE_HEIGHT:
        return None
    return float(x1), float(x2), float(y1), float(y2)


def make_detections(pos, vel, cams, rng, noise_px=4.0, centroid_noise_px=6.0):
    dims = DEFAULT_DIMENSIONS.as_array()
    corners_all = box_corners(pos, vel, dims)
    rows = {cam: [] for cam in cams}
    for f in range(NUM_FRAMES):
        for cam, cam_list in cams.items():
            c = cam_list[f]
            # сценарии потери видимости
            if 200 <= f <= 300 and cam != 3:
                continue                                    # видит только Cam3
            if 500 <= f <= 514:
                continue                                    # никто не видит
            if cam == 2 and 330 <= f <= 360:
                continue                                    # Cam2 за препятствием
            if cam == 4 and f >= 812:
                continue                                    # Cam4 потеряла после заезда за препятствие
            box = visible_box(c, corners_all[f])
            if box is None:
                continue
            x1, x2, y1, y2 = box
            width = x2 - x1
            if width < 12:
                continue
            x1 += rng.normal(0, noise_px)
            x2 += rng.normal(0, noise_px)
            y1 += rng.normal(0, noise_px)
            y2 += rng.normal(0, noise_px)
            if cam == 4 and 790 <= f < 812:
                x2 -= 0.045 * width * (f - 789)             # правое ребро «съёживается» за препятствием
            conf = float(np.clip(rng.normal(0.85, 0.08), 0.3, 0.99))
            outlier = rng.random() < 0.01
            if outlier:                                     # ложная детекция: рамка в другом месте кадра
                x1, x2 = x1 + rng.uniform(-400, 400), x2 + rng.uniform(-400, 400)
                x1, x2 = min(x1, x2), max(x1, x2)
                conf = float(rng.uniform(0.3, 0.6))
            x1, x2 = np.clip([x1, x2], 0, IMAGE_WIDTH)
            y1, y2 = np.clip([y1, y2], 0, IMAGE_HEIGHT)
            center_uv = c.project(corners_all[f].mean(axis=0))
            cu = float(center_uv[0] + rng.normal(0, centroid_noise_px))
            cv = float(center_uv[1] + rng.normal(0, centroid_noise_px))
            if outlier:
                cu, cv = (x1 + x2) / 2, (y1 + y2) / 2
            area = float(max((x2 - x1) * (y2 - y1) * 0.6, 1.0))
            m = 2.0
            edge_clipped = bool(x1 <= m or y1 <= m or x2 >= IMAGE_WIDTH - m or y2 >= IMAGE_HEIGHT - m)
            too_small = area < 400.0
            unreliable = edge_clipped or too_small or conf < 0.5
            rows[cam].append(Detection(cam, f, conf, float(x1), float(y1), float(x2), float(y2), cu, cv,
                                       "mask_centroid", area, cu, cv, num_candidates=1 + int(outlier),
                                       edge_clipped=edge_clipped, too_small=too_small, unreliable=unreliable))
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("tmp/synthetic"))
    p.add_argument("--out", type=Path, default=Path("outputs/synthetic"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    frames = np.arange(NUM_FRAMES, dtype=float)
    pos, vel = trajectory(frames)
    cams = camera_poses(pos)
    args.data.mkdir(parents=True, exist_ok=True)
    for cam, cam_list in cams.items():
        write_camera_csv(camera_csv_path(args.data, cam), cam_list, FOCAL_MM[cam])
    write_target_csv(target_csv_path(args.data), pos)

    det_dir = args.out / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    rows = make_detections(pos, vel, cams, rng)
    for cam, dets in rows.items():
        with open(det_dir / f"cam{cam}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=Detection.columns())
            w.writeheader()
            w.writerows(d.to_row() for d in dets)
        np.savez_compressed(det_dir / f"cam{cam}_polygons.npz")
    (det_dir / "meta.json").write_text(json.dumps({"weights": "synthetic", "note": "синтетика, без YOLO"},
                                                  ensure_ascii=False, indent=2))
    speed = np.linalg.norm(vel, axis=1) * FPS / 100
    print(f"данные: {args.data}, детекции: {det_dir}; скорость {speed.min():.1f}–{speed.max():.1f} м/с; "
          + ", ".join(f"cam{c}: {len(d)} кадров" for c, d in rows.items()))


if __name__ == "__main__":
    main()
