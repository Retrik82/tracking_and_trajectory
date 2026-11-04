"""
Точность без эталона: проверка на отложенной камере.

Трекер (потоковый конвейер, режим detections) считается БЕЗ камеры c; его оценка (центр, курс, габариты)
проецируется в камеру c и сравнивается с её рамкой YOLO на кадрах с надёжной детекцией:
  ошибка центра рамки, px, и в метрах (px · глубина / f_x) — насколько оценка положения промахивается мимо
  грузовика, которого эта камера реально видит. Камера c в оценку не входила — это честная проверка.
Систематика коробки (форма ≠ грузовик) одинакова для всех вариантов — сравнение вариантов корректно,
абсолютные значения — верхняя оценка ошибки.

python tools/heldout_eval.py --data data --detections outputs/scene2_all/detections --holdout 2,6,9 \
    --set tracker.box_length_m=21.3 [--set ...]
"""
import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pipeline.config import StreamConfig  # noqa: E402
from pipeline.streaming import StreamPipeline, load_detections_table, load_polygons  # noqa: E402
from reconstruction.data_io import detections_image_sizes, make_camera  # noqa: E402
from reconstruction.fusion import silhouette_points  # noqa: E402


def evaluate(cfg: StreamConfig, data: Path, det_dir: Path, holdout: list[int], cams: list[int]) -> dict:
    sizes = detections_image_sizes(det_dir)
    table, n = load_detections_table(det_dir, cams)
    need_masks = cfg.orientation.enabled or cfg.silhouette.enabled
    polys = load_polygons(det_dir, cams) if need_masks else None
    px, m, per_cam = [], [], {}
    for c in holdout:
        run = cfg.copy()
        run.cameras.ids = [x for x in cams if x != c]
        run.output.plots = run.output.performance_report = run.output.save_track_csv = False
        run.streaming.mode, run.streaming.realtime_pacing, run.streaming.log_every = "detections", False, 0
        with tempfile.TemporaryDirectory() as tmp:
            pipe = StreamPipeline(run, data, Path(tmp), {k: v for k, v in table.items() if k != c}, n,
                                  image_sizes=sizes, polygons=polys)
            records = pipe.run()
        dims = pipe.dims.as_array()
        e_px, e_m = [], []
        for r in records:
            d = table[c].get(r.index)
            if d is None or d.unreliable or not np.all(np.isfinite(r.box_center)) or not np.isfinite(r.heading_deg):
                continue
            cam = make_camera(data, c, r.index + run.cameras.frame_offset, sizes[c])
            h = np.radians(r.heading_deg)
            pts = cam.project(silhouette_points(r.box_center, np.array([np.cos(h), np.sin(h), 0.0]),
                                                pipe.model, dims))
            if not np.all(np.isfinite(pts)):
                continue
            est = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2])
            obs = np.array([(d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2])
            err = float(np.linalg.norm(est - obs))
            e_px.append(err)
            e_m.append(err * cam.depth(r.box_center) / cam.f_x / run.cameras.units_per_meter)
        per_cam[c] = (len(e_px), float(np.median(e_m)) if e_m else float("nan"))
        px += e_px
        m += e_m
    px, m = np.array(px), np.array(m)
    return {"frames": len(px), "px_p50": float(np.median(px)), "px_p95": float(np.percentile(px, 95)),
            "m_p50": float(np.median(m)), "m_p95": float(np.percentile(m, 95)), "per_cam": per_cam}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--holdout", default="2,6,9")
    p.add_argument("--cameras", default="1,2,3,4,5,6,7,8,9")
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--name", default="")
    a = p.parse_args()
    logging.basicConfig(level=logging.ERROR)
    cfg = StreamConfig.load(a.config, a.set)
    r = evaluate(cfg, a.data, a.detections, [int(x) for x in a.holdout.split(",")], [int(x) for x in a.cameras.split(",")])
    cams = "  ".join(f"cam{c}: {k} кадров, {v:.2f} м" for c, (k, v) in r["per_cam"].items())
    print(f"{a.name}: кадров {r['frames']}, ошибка на отложенной камере p50/p95 {r['m_p50']:.2f} / {r['m_p95']:.2f} м "
          f"({r['px_p50']:.0f} / {r['px_p95']:.0f} px) | {cams}", flush=True)


if __name__ == "__main__":
    main()
