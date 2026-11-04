"""
Проверка геометрии на реальных данных сцены 2 (без эталона):
1) триангуляция центров масок двух камер и ошибка перепроецирования по кадрам;
2) наложение оценки трекера (коробка) и рамки детектора на настоящий кадр видео.

python real_check.py <repo> <data> <detections> <stream_track.csv> <out_dir> cams frames
"""
import csv
import sys
from pathlib import Path

repo, data, det_dir, track_csv, out_dir = (Path(a) for a in sys.argv[1:6])
cams = [int(c) for c in sys.argv[6].split(",")]
frames = [int(f) for f in sys.argv[7].split(",")]
sys.path.insert(0, str(repo))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from detection.schema import Detection  # noqa: E402
from reconstruction.data_io import find_video, make_camera, probe_video_resolution  # noqa: E402
from reconstruction.fusion import silhouette_points  # noqa: E402
from reconstruction.triangulation import triangulate_robust  # noqa: E402

sizes = {c: probe_video_resolution(find_video(data, c)) for c in cams}
dets = {}
for c in cams:
    with open(det_dir / f"cam{c}.csv", newline="") as f:
        dets[c] = {d.frame: d for d in map(Detection.from_row, csv.DictReader(f))}

# 1) перепроецирование
errs, n = {c: [] for c in cams}, 0
for f in sorted(set.intersection(*(set(dets[c]) for c in cams))):
    ds = [dets[c][f] for c in cams]
    if any(d.unreliable for d in ds):
        continue
    res = triangulate_robust([make_camera(data, c, f, sizes[c]) for c in cams],
                             [(d.point_u, d.point_v) for d in ds], max_reprojection_error_px=1e9)
    if res.ok:
        n += 1
        for c, e in zip(cams, res.reprojection_errors):
            errs[c].append(e)
print(f"кадров с надёжной стерео-парой {cams}: {n}")
for c in cams:
    e = np.array(errs[c])
    if len(e):
        print(f"  cam{c}: ошибка перепроецирования центра маски, px: медиана {np.median(e):.1f}, p90 {np.percentile(e, 90):.1f}")

# 2) наложение
rows = {int(r["frame"]): r for r in csv.DictReader(open(track_csv, encoding="utf-8"))}
dims = np.array([1650.0, 255.0, 400.0])
out_dir.mkdir(parents=True, exist_ok=True)
for c in cams:
    cap = cv2.VideoCapture(str(find_video(data, c)))
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        cam = make_camera(data, c, f, sizes[c])
        d = dets[c].get(f)
        if d is not None:
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (200, 200, 200), 2)
        r = rows.get(f)
        label = f"cam{c} frame {f} {img.shape[1]}x{img.shape[0]} f={cam.focal_length_mm:g}mm fx={cam.f_x:.0f}px"
        if r is not None and r["x"] not in ("nan", ""):
            center = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
            h = np.radians(float(r["heading_deg"]))
            pts = cam.project(silhouette_points(center, np.array([np.cos(h), np.sin(h), 0.0]), None, dims))
            pts = pts[np.all(np.isfinite(pts), axis=1)].astype(np.int32)
            if len(pts) > 2:
                cv2.polylines(img, [cv2.convexHull(pts)], True, (214, 120, 42), 3, cv2.LINE_AA)
            label += f"  {r['status']}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (20, 20, 20), -1)
        cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        path = out_dir / f"cam{c}_f{f}.jpg"
        cv2.imwrite(str(path), cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
        print(path)
    cap.release()
