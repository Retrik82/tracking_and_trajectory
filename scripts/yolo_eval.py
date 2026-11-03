"""
Проверка точности 3D-траектории грузовика по реальным детекциям YOLO.

Шаги (каждый сохраняет результат на диск, повторный запуск переиспользует детекции):
  detect     — YOLO по всем видео -> detections/cam{N}.csv + контуры масок
  evaluate   — триангуляция, B-сплайн, сравнение с CSV цели, графики, report.md
  video      — видео с наложением по каждой камере (флаг --video)

Пример:
  .venv/bin/python scripts/yolo_eval.py --weights models/best.pt --data tmp/project2 --video
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.schema import Detection  # noqa: E402
from reconstruction.data_io import video_path  # noqa: E402

CAMERAS = [1, 2, 3, 4, 5]
log = logging.getLogger("yolo_eval")


# ---------------------------------------------------------------- детекция

def detections_dir(out: Path) -> Path:
    return out / "detections"


def run_detection(args) -> None:
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
        with open(detections_dir(out) / f"cam{cam}.csv", newline="") as f:
            result[cam] = {d.frame: d for d in map(Detection.from_row, csv.DictReader(f))}
    return result


def load_polygons(out: Path, cam: int) -> dict[int, np.ndarray]:
    path = detections_dir(out) / f"cam{cam}_polygons.npz"
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {int(k): data[k] for k in data.files}


# ---------------------------------------------------------------- запуск

def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    p.add_argument("--data", type=Path, default=Path("tmp/project2"))
    p.add_argument("--out", type=Path, default=Path("outputs/yolo_eval"))
    p.add_argument("--steps", default="detect,evaluate", help="через запятую: detect, evaluate, video")
    p.add_argument("--video", action="store_true", help="добавить шаг video")
    p.add_argument("--redetect", action="store_true", help="пересчитать детекции, даже если они сохранены")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default=None)
    p.add_argument("--min-mask-area", type=float, default=400.0, help="маска меньше — флаг too_small, px")
    p.add_argument("--min-reliable-conf", type=float, default=0.5)
    p.add_argument("--frame-offset", type=int, default=0, help="кадр CSV = кадр видео + смещение")
    p.add_argument("--max-reproj-px", type=float, default=20.0, help="порог отбрасывания камеры в триангуляции")
    p.add_argument("--knot-spacing", type=float, default=15.0, help="шаг узлов сплайна, кадры")
    p.add_argument("--huber-scale", type=float, default=100.0, help="порог Huber для сплайна, единицы мира")
    p.add_argument("--units-per-meter", type=float, default=100.0, help="единицы мира в метре (UE: см)")
    p.add_argument("--fps", type=float, default=30.0)
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", encoding="utf-8")],
    )


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.out)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if args.video and "video" not in steps:
        steps.append("video")

    if "detect" in steps:
        run_detection(args)
    if "evaluate" in steps or "video" in steps:
        from scripts.yolo_analysis import run_evaluation, run_video

        result = run_evaluation(args, load_detections(args.out))
        if "video" in steps:
            run_video(args, result, load_polygons)


if __name__ == "__main__":
    main()
