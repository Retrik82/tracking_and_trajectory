"""
Экспорт координат трекера в CSV (trajectory.csv) — только чтение готовых результатов, алгоритм не меняется.

Колонки:
  frame_id    номер кадра видео
  timestamp   время кадра, с (кадр / FPS в replay/detections; часы захвата в live)
  object_id   номер объекта (сопровождается один грузовик — всегда 1)
  x, y, z     центр кабины грузовика (основная точка вывода трекера реального времени), МЕТРЫ,
              мировая система Unreal Engine: левая, Z вверх (единицы сцены — см, делятся на units_per_meter)
  camera_id   камеры, чьи детекции вошли в оценку на этом кадре, через пробел; пусто — кадр без измерений
              (оценка — прогноз фильтра, status = predicted)
  status      measured / partial / predicted (кадры инициализации без координат не экспортируются)
  position_std_m  СКО положения по ковариации фильтра, м

Значения берутся из тех же объектов, что stream_track.csv (TickRecord) и realtime_track.npz (TrackResult):
никаких детекций, интерполяции или «будущих» кадров.
"""

import csv
from pathlib import Path

import numpy as np

COLUMNS = ["frame_id", "timestamp", "object_id", "x", "y", "z", "camera_id", "status", "position_std_m"]
OBJECT_ID = 1


def write_trajectory_csv(path: Path, rows) -> int:
    """rows — (кадр, время с, статус, центр кабины (3,) в метрах, камеры, СКО м); возвращает число строк."""
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for frame, ts, status, xyz_m, cams, std_m in rows:
            if not np.all(np.isfinite(xyz_m)):
                continue                                   # инициализация: координат ещё нет
            w.writerow([int(frame), f"{ts:.4f}", OBJECT_ID, f"{xyz_m[0]:.4f}", f"{xyz_m[1]:.4f}", f"{xyz_m[2]:.4f}",
                        " ".join(str(int(c)) for c in cams), status,
                        f"{std_m:.4f}" if np.isfinite(std_m) else ""])
            n += 1
    return n


def stream_rows(records, upm: float):
    """Из TickRecord потокового конвейера."""
    for r in records:
        yield (r.index, r.timestamp, r.status, np.asarray(r.cab_center, float) / upm,
               r.detected_cameras if r.status != "predicted" else [], r.position_std_m)


def tracker_rows(result, frames, fps: float, upm: float, cams_by_frame: dict):
    """Из TrackResult трекера main.py (step_realtime); cams_by_frame — камеры с детекцией на кадре."""
    for frame, e in zip(frames, result.estimates):
        std = e.position_std / upm if np.isfinite(e.position_std) else float("nan")
        cams = [] if e.status == "predicted" else sorted(set(cams_by_frame.get(int(frame), [])) - set(e.rejected_cameras))
        yield (frame, frame / fps, e.status, np.asarray(e.cab_center, float) / upm, cams, std)
