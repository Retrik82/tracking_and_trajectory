"""
Синтетическая сцена в формате проекта: CSV камер, CSV цели и «детекции» по коробке грузовика.

Нужна, чтобы прогнать конвейер (кроме detect и video) без видео, весов YOLO и 3D-модели:
проверить код, посмотреть отчёт и графики, сравнить методы на данных с известной правдой,
прогнать обязательные сценарии ТЗ (число камер, перекрытия, пропадание, выбросы, рассинхронизация).
Это НЕ замена реальным данным: цифры на синтетике ничего не говорят о точности на видео.

Базовая сцена (scenario = base), похожая на реальную (см. README):
- грузовик — коробка DEFAULT_DIMENSIONS (16.5 × 2.55 × 4 м), едет по петле с переменной
  скоростью (до ~25 м/с, продольное ускорение до ~5 м/с², поперечное до ~6 м/с²);
- точка CSV цели — низ передней грани коробки (то, что трекер выдаёт как position);
- 5 камер с фокусами 35/60/35/50/20 мм; Camera5 поворачивается по yaw на части кадров;
- детекции: рамка проекции коробки с шумом, центр маски — проекция центра коробки с шумом,
  уверенность случайная; участки потери видимости: кадры 200–300 видит одна камера,
  кадры 500–514 не видит никто, Cam4 «заезжает за препятствие» (ребро съёживается) и пропадает,
  редкие ложные детекции.

Сценарии (--scenario, см. SCENARIOS): cams4 / cams3 / cams2 / cam1, occlusion, blackout, reappear,
sharp_turn, accel_brake, speed_change, false_detection, outlier, desync, handover.

Запуск:
    python tools/make_synthetic_data.py --data tmp/synthetic --out outputs/synthetic
    python tools/make_synthetic_data.py --scenario cams2 --data tmp/scen/cams2 --out outputs/scen/cams2
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
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


@dataclass
class Scenario:
    name: str = "base"
    cameras: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])   # камеры с детекциями (остальные — пустой CSV)
    single_camera_segment: tuple[int, int, int] | None = (200, 300, 3)    # (от, до, единственная камера)
    blackouts: list[tuple[int, int]] = field(default_factory=lambda: [(500, 514)])   # никто не видит
    occlusions: list[tuple[int, int, int]] = field(default_factory=lambda: [(2, 330, 360)])   # (камера, от, до)
    shrink: tuple[int, int, int] | None = (4, 790, 812)                  # камера, ребро съёживается, затем пропадает
    false_rate: float = 0.01                                             # ложная рамка (в другом месте кадра)
    centroid_outlier_rate: float = 0.0                                   # центроид маски прыгает при верной рамке
    noise_px: float = 4.0
    centroid_noise_px: float = 6.0
    speed_mean: float = 1600.0                                           # см/с
    speed_amplitude: float = 800.0
    speed_period_s: float = 15.0
    ellipse: tuple[float, float] = (12000.0, 7000.0)                     # полуоси, см (меньше — круче повороты)
    stop: tuple[int, int] | None = None                                  # остановка (кадры): скорость → 0
    source_delays: dict = field(default_factory=dict)                    # рассинхронизация, кадров (режим detections)
    overrides: dict = field(default_factory=dict)                        # параметры конфигурации для этого сценария


SCENARIOS = {
    "base": Scenario(),
    "cams5": Scenario("cams5", single_camera_segment=None, blackouts=[], occlusions=[], shrink=None, false_rate=0.0),
    "cams4": Scenario("cams4", cameras=[1, 2, 3, 5], single_camera_segment=None, blackouts=[], occlusions=[], shrink=None,
                      false_rate=0.0),
    "cams3": Scenario("cams3", cameras=[1, 3, 5], single_camera_segment=None, blackouts=[], occlusions=[], shrink=None,
                      false_rate=0.0),
    "cams2": Scenario("cams2", cameras=[1, 3], single_camera_segment=None, blackouts=[], occlusions=[], shrink=None,
                      false_rate=0.0),
    "cam1": Scenario("cam1", cameras=[3], single_camera_segment=None, blackouts=[], occlusions=[], shrink=None,
                     false_rate=0.0),
    "cam1_mono": Scenario("cam1_mono", cameras=[3], single_camera_segment=None, blackouts=[], occlusions=[], shrink=None,
                          false_rate=0.0, overrides={"tracker.mono_init": True}),
    "occlusion": Scenario("occlusion", single_camera_segment=None, blackouts=[],
                          occlusions=[(1, 300, 360), (3, 320, 380), (5, 600, 640)], shrink=(4, 790, 812), false_rate=0.0),
    "blackout": Scenario("blackout", single_camera_segment=None, blackouts=[(400, 430), (700, 760)], occlusions=[],
                         shrink=None, false_rate=0.0),
    "reappear": Scenario("reappear", single_camera_segment=None, blackouts=[(450, 470), (480, 500), (520, 560)],
                         occlusions=[], shrink=None, false_rate=0.0),
    "sharp_turn": Scenario("sharp_turn", ellipse=(9000.0, 4000.0), single_camera_segment=None, blackouts=[], occlusions=[],
                           shrink=None, false_rate=0.0),
    "accel_brake": Scenario("accel_brake", speed_mean=1500.0, speed_amplitude=1300.0, speed_period_s=8.0,
                            single_camera_segment=None, blackouts=[], occlusions=[], shrink=None, false_rate=0.0),
    "speed_change": Scenario("speed_change", stop=(450, 540), single_camera_segment=None, blackouts=[], occlusions=[],
                             shrink=None, false_rate=0.0),
    "false_detection": Scenario("false_detection", false_rate=0.08, single_camera_segment=None, blackouts=[], occlusions=[],
                                shrink=None),
    "outlier": Scenario("outlier", centroid_outlier_rate=0.05, single_camera_segment=None, blackouts=[], occlusions=[],
                        shrink=None, false_rate=0.0),
    "desync": Scenario("desync", source_delays={2: 1, 4: 2}, single_camera_segment=None, blackouts=[], occlusions=[],
                       shrink=None, false_rate=0.0),
    "handover": Scenario("handover", single_camera_segment=(150, 330, 1), blackouts=[], occlusions=[], shrink=None,
                         false_rate=0.0),
}


def trajectory(frames: np.ndarray, sc: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Опорная точка (низ передней грани), см, и скорость, см/кадр. Петля с переменной скоростью."""
    t = frames / FPS
    speed = sc.speed_mean + sc.speed_amplitude * np.sin(2 * np.pi * t / sc.speed_period_s)
    speed = np.maximum(speed, 100.0)
    if sc.stop is not None:
        a, b = sc.stop
        ramp = 15
        for i, f in enumerate(frames):
            if a - ramp <= f < a:
                speed[i] *= (a - f) / ramp
            elif a <= f <= b:
                speed[i] = 0.0
            elif b < f <= b + ramp:
                speed[i] *= (f - b) / ramp
    dt = 1.0 / FPS
    path = np.concatenate([[0.0], np.cumsum((speed[1:] + speed[:-1]) / 2 * dt)])
    a, b = sc.ellipse
    perimeter = 2 * np.pi * np.sqrt((a ** 2 + b ** 2) / 2)
    theta = 2 * np.pi * path / perimeter
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    z = 300.0 + 150.0 * np.sin(theta) + 40.0 * np.sin(3 * theta)      # лёгкий рельеф
    pos = np.c_[x, y, z]
    vel = np.gradient(pos, axis=0)
    # при остановке скорость нулевая, а курс (ориентация коробки) сохраняется — направление из касательной к пути
    tangent = np.c_[-a * np.sin(theta), b * np.cos(theta), np.gradient(z)]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    speed_frame = np.linalg.norm(vel, axis=1, keepdims=True)
    direction = tangent * np.maximum(speed_frame, 1.0)        # модуль не меньше 1 см/кадр: оси коробки определены
    return pos, direction


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


def make_detections(pos, vel, cams, rng, sc: Scenario):
    dims = DEFAULT_DIMENSIONS.as_array()
    corners_all = box_corners(pos, vel, dims)
    rows = {cam: [] for cam in cams}
    for f in range(NUM_FRAMES):
        for cam, cam_list in cams.items():
            if cam not in sc.cameras:
                continue
            c = cam_list[f]
            # сценарии потери видимости
            if sc.single_camera_segment and sc.single_camera_segment[0] <= f <= sc.single_camera_segment[1] \
                    and cam != sc.single_camera_segment[2]:
                continue
            if any(a <= f <= b for a, b in sc.blackouts):
                continue
            if any(cam == oc and a <= f <= b for oc, a, b in sc.occlusions):
                continue
            if sc.shrink and cam == sc.shrink[0] and f >= sc.shrink[2]:
                continue                                    # потеряла после заезда за препятствие
            box = visible_box(c, corners_all[f])
            if box is None:
                continue
            x1, x2, y1, y2 = box
            width = x2 - x1
            if width < 12:
                continue
            x1 += rng.normal(0, sc.noise_px)
            x2 += rng.normal(0, sc.noise_px)
            y1 += rng.normal(0, sc.noise_px)
            y2 += rng.normal(0, sc.noise_px)
            if sc.shrink and cam == sc.shrink[0] and sc.shrink[1] <= f < sc.shrink[2]:
                x2 -= 0.045 * width * (f - sc.shrink[1] + 1)   # правое ребро «съёживается» за препятствием
            conf = float(np.clip(rng.normal(0.85, 0.08), 0.3, 0.99))
            outlier = rng.random() < sc.false_rate
            if outlier:                                     # ложная детекция: рамка в другом месте кадра
                x1, x2 = x1 + rng.uniform(-400, 400), x2 + rng.uniform(-400, 400)
                x1, x2 = min(x1, x2), max(x1, x2)
                conf = float(rng.uniform(0.3, 0.6))
            x1, x2 = np.clip([x1, x2], 0, IMAGE_WIDTH)
            y1, y2 = np.clip([y1, y2], 0, IMAGE_HEIGHT)
            center_uv = c.project(corners_all[f].mean(axis=0))
            cu = float(center_uv[0] + rng.normal(0, sc.centroid_noise_px))
            cv = float(center_uv[1] + rng.normal(0, sc.centroid_noise_px))
            if outlier:
                cu, cv = (x1 + x2) / 2, (y1 + y2) / 2
            elif rng.random() < sc.centroid_outlier_rate:   # выброс в координатах точки при верной рамке
                cu += rng.uniform(-150, 150)
                cv += rng.uniform(-80, 80)
            area = float(max((x2 - x1) * (y2 - y1) * 0.6, 1.0))
            m = 2.0
            edge_clipped = bool(x1 <= m or y1 <= m or x2 >= IMAGE_WIDTH - m or y2 >= IMAGE_HEIGHT - m)
            too_small = area < 400.0
            unreliable = edge_clipped or too_small or conf < 0.5
            rows[cam].append(Detection(cam, f, conf, float(x1), float(y1), float(x2), float(y2), cu, cv,
                                       "mask_centroid", area, cu, cv, num_candidates=1 + int(outlier),
                                       edge_clipped=edge_clipped, too_small=too_small, unreliable=unreliable))
    return rows


def generate(data_dir: Path, out_dir: Path, sc: Scenario, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    frames = np.arange(NUM_FRAMES, dtype=float)
    pos, vel = trajectory(frames, sc)
    cams = camera_poses(pos)
    data_dir.mkdir(parents=True, exist_ok=True)
    for cam, cam_list in cams.items():
        write_camera_csv(camera_csv_path(data_dir, cam), cam_list, FOCAL_MM[cam])
    write_target_csv(target_csv_path(data_dir), pos)
    det_dir = out_dir / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    rows = make_detections(pos, vel, cams, rng, sc)
    for cam, dets in rows.items():
        with open(det_dir / f"cam{cam}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=Detection.columns())
            w.writeheader()
            w.writerows(d.to_row() for d in dets)
        np.savez_compressed(det_dir / f"cam{cam}_polygons.npz")
    (det_dir / "meta.json").write_text(json.dumps({"weights": "synthetic", "scenario": sc.name,
                                                   "source_delays": sc.source_delays,
                                                   "note": "синтетика, без YOLO"}, ensure_ascii=False, indent=2))
    speed = np.linalg.norm(vel, axis=1) * FPS / 100
    return {"scenario": sc.name, "speed_min": float(speed.min()), "speed_max": float(speed.max()),
            "detections": {c: len(d) for c, d in rows.items()}, "source_delays": sc.source_delays}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("tmp/synthetic"))
    p.add_argument("--out", type=Path, default=Path("outputs/synthetic"))
    p.add_argument("--scenario", default="base", choices=sorted(SCENARIOS))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    info = generate(args.data, args.out, SCENARIOS[args.scenario], args.seed)
    print(f"сценарий {info['scenario']}: данные {args.data}, детекции {args.out / 'detections'}; "
          f"скорость {info['speed_min']:.1f}–{info['speed_max']:.1f} м/с; "
          + ", ".join(f"cam{c}: {n} кадров" for c, n in info["detections"].items()))


if __name__ == "__main__":
    main()
