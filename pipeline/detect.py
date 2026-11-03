"""
Шаг detect: YOLO по всем пяти видео, детекции сохраняются на диск.

Результат в <out>/detections/:
  cam{N}.csv            одна строка на кадр, где есть грузовик (формат — detection.schema.Detection)
  cam{N}_polygons.npz   контур маски выбранного объекта по кадрам (ключ — номер кадра)
  meta.json             веса, imgsz, порог уверенности, устройство, классы модели

Повторный запуск пропускает камеры, для которых CSV уже есть (--redetect — пересчитать).
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from detection.schema import Detection
from pipeline.center import CAMERAS
from reconstruction.data_io import video_path

log = logging.getLogger("peleng")


def detections_dir(out: Path) -> Path:
    return out / "detections"


def run_detection(args) -> None:
    # ultralytics импортируется только здесь: остальные шаги работают по сохранённым детекциям без него
    from detection.yolo import WeightsNotFoundError, YoloDetector, YoloSettings

    try:
        detector = YoloDetector(YoloSettings(
            weights_path=args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device,
            min_mask_area_px=args.min_mask_area, min_reliable_conf=args.min_reliable_conf,
        ))
    except WeightsNotFoundError as e:
        log.error("%s", e)
        sys.exit(2)
    log.info("модель %s, задача %s, устройство %s", args.weights, detector.model.task, detector.device)

    out = detections_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        csv_path = out / f"cam{cam}.csv"
        if csv_path.exists() and not args.redetect:
            log.info("cam%d: детекции уже есть (%s), пропускаю; --redetect чтобы пересчитать", cam, csv_path)
            continue
        started = time.time()
        rows, polygons = [], {}
        for res in detector.detect_video(video_path(args.data, cam), cam):
            if res.detection is None:
                continue
            rows.append(res.detection.to_row())
            if res.polygon is not None and len(res.polygon):
                polygons[str(res.detection.frame)] = res.polygon
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=Detection.columns())
            writer.writeheader()
            writer.writerows(rows)
        # Маски пока не используются дальше (трекер работает по рамкам), но сохраняются, чтобы
        # перейти на сравнение силуэта 3D-модели с маской без повторной детекции
        np.savez_compressed(out / f"cam{cam}_polygons.npz", **polygons)
        log.info("cam%d: %d кадров с детекцией за %.0f с", cam, len(rows), time.time() - started)

    (out / "meta.json").write_text(json.dumps({
        "weights": str(args.weights), "imgsz": args.imgsz, "conf": args.conf,
        "device": detector.device, "classes": detector.model.names,
    }, ensure_ascii=False, indent=2))


def load_detections(out: Path) -> dict[int, dict[int, Detection]]:
    """{камера: {кадр видео: Detection}}"""
    result = {}
    for cam in CAMERAS:
        path = detections_dir(out) / f"cam{cam}.csv"
        if not path.exists():
            raise SystemExit(f"Нет детекций {path}: запустите шаг detect")
        with open(path, newline="") as f:
            result[cam] = {d.frame: d for d in map(Detection.from_row, csv.DictReader(f))}
    return result
