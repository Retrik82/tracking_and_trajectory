"""
Шаг detect: YOLO по всем пяти видео, детекции сохраняются на диск.

Результат в <out>/detections/:
  cam{N}.csv            одна строка на кадр, где есть грузовик (формат — detection.schema.Detection)
  cam{N}_polygons.npz   контур маски выбранного объекта по кадрам (ключ — номер кадра)
  meta.json             веса, imgsz, порог уверенности, устройство, классы модели; image_size —
                        разрешение видео каждой камеры, на котором сделаны детекции (для шагов без видео)

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
from reconstruction.data_io import (VideoResolutionError, detections_image_sizes, find_video,
                                    probe_video_resolution, video_path, write_detections_image_sizes)

log = logging.getLogger("peleng")


def detections_dir(out: Path) -> Path:
    return out / "detections"


def _video_size(args, cam: int) -> tuple[int, int]:
    """Разрешение видео камеры перед детекцией; не определить — понятная ошибка, а не детекция «вслепую»."""
    try:
        return probe_video_resolution(video_path(args.data, cam))
    except VideoResolutionError as e:
        raise SystemExit(f"cam{cam}: {e}") from None


def _record_image_sizes(args, out: Path, previous: dict[int, tuple[int, int]]) -> None:
    """
    Разрешение видео по камерам — в meta.json: шаги без видео (center, cab, realtime, stream detections)
    строят внутренние параметры камер по нему. Камера без видео сохраняет прежнее значение.
    """
    sizes = dict(previous)
    for cam in CAMERAS:
        if find_video(args.data, cam) is not None:
            sizes[cam] = _video_size(args, cam)
    write_detections_image_sizes(out, sizes)


def run_detection(args) -> None:
    if getattr(args, "detector", "yolo") == "classic":
        run_classic_detection(args)
        return
    # ultralytics импортируется только здесь: остальные шаги работают по сохранённым детекциям без него
    from detection.yolo import WeightsNotFoundError, YoloDetector, YoloSettings

    try:
        detector = YoloDetector(YoloSettings(
            weights_path=args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device,
            backend=getattr(args, "backend", "auto"),
            min_mask_area_px=args.min_mask_area, min_reliable_conf=args.min_reliable_conf,
        ))
    except WeightsNotFoundError as e:
        log.error("%s", e)
        sys.exit(2)
    log.info("модель %s, задача %s, устройство %s", args.weights, detector.model.task, detector.device)

    out = detections_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    previous_sizes = detections_image_sizes(out)
    for cam in CAMERAS:
        csv_path = out / f"cam{cam}.csv"
        if csv_path.exists() and not args.redetect:
            log.info("cam%d: детекции уже есть (%s), пропускаю; --redetect чтобы пересчитать", cam, csv_path)
            continue
        log.info("cam%d: %s, разрешение %d×%d", cam, video_path(args.data, cam).name, *_video_size(args, cam))
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
        "device": detector.device, "backend": detector.backend, "classes": detector.model.names,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _record_image_sizes(args, out, previous_sizes)


def _save_camera(out: Path, cam: int, results) -> int:
    rows, polygons = [], {}
    for res in results:
        if res.detection is None:
            continue
        rows.append(res.detection.to_row())
        if res.polygon is not None and len(res.polygon):
            polygons[str(res.detection.frame)] = res.polygon
    with open(out / f"cam{cam}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=Detection.columns())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / f"cam{cam}_polygons.npz", **polygons)
    return len(rows)


def camera_rotations(data_dir: Path, cam: int, frame_offset: int = 0) -> tuple[np.ndarray, list | None]:
    """K полного кадра и повороты мир→камера по кадрам видео (None — камера не поворачивается)."""
    from reconstruction.camera import load_camera_poses
    from reconstruction.data_io import camera_csv_path, make_camera

    camera = make_camera(data_dir, cam, frame_offset)        # разрешение — из видео этой камеры
    poses = load_camera_poses(str(camera_csv_path(data_dir, cam)))
    angles = {f: p[1:] for f, p in poses.items()}
    if len({tuple(a) for a in angles.values()}) <= 1:
        return camera.K, None
    rotations, last = [], camera.R
    for frame in range(max(poses) - frame_offset + 1):
        a = angles.get(frame + frame_offset)
        if a is not None:
            camera.update_rotation(*a)
            last = camera.R.copy()
        rotations.append(last)
    return camera.K, rotations


def run_classic_detection(args) -> None:
    """Детекции без нейросети: вычитание фона с учётом известного поворота камеры."""
    from detection.classic import ClassicDetector, ClassicSettings

    settings = ClassicSettings(min_mask_area_px=args.min_mask_area)
    detector = ClassicDetector(settings)
    out = detections_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    previous_sizes = detections_image_sizes(out)
    for cam in CAMERAS:
        if (out / f"cam{cam}.csv").exists() and not args.redetect:
            log.info("cam%d: детекции уже есть, пропускаю; --redetect чтобы пересчитать", cam)
            continue
        log.info("cam%d: %s, разрешение %d×%d", cam, video_path(args.data, cam).name, *_video_size(args, cam))
        started = time.time()
        K, rotations = camera_rotations(args.data, cam, args.frame_offset)
        n = _save_camera(out, cam, detector.detect_video(video_path(args.data, cam), cam, rotations, K))
        log.info("cam%d: %d кадров с детекцией за %.0f с (без нейросети%s)", cam, n, time.time() - started,
                 ", с компенсацией поворота" if rotations is not None else "")
    (out / "meta.json").write_text(json.dumps({
        "detector": "classic (вычитание фона, detection/classic.py)",
        "settings": {k: v for k, v in vars(settings).items()},
    }, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    _record_image_sizes(args, out, previous_sizes)


def detector_info(out: Path) -> dict:
    """Чем получены детекции (из meta.json): для подписей в отчётах и видео."""
    try:
        meta = json.loads((detections_dir(out) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if str(meta.get("detector", "")).startswith("classic"):
        return {"name": "classic", "source": "детектор без нейросети (вычитание фона)",
                "note": "без нейросети (вычитание фона)", "note_en": "no neural net (background subtraction)",
                "warning": "Детекции без нейросети (вычитание фона); параметры детектора подобраны на этих же видео."}
    return {"name": "yolo", "source": f"веса `{meta.get('weights', '?')}`",
            "note": "YOLO обучена на этих же видео", "note_en": "YOLO trained on these videos",
            "warning": "Модель дообучена на этих же видео: точность детекции оценена на обучающих данных."}


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
