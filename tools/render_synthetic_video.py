"""
Рендер синтетических видео пяти камер для демонстрации потокового конвейера без реальных записей.

Сцена — та же, что в tools/make_synthetic_data.py (тот же сценарий, те же позы камер и траектория):
рисуются сетка дороги, коробка грузовика (грани с простой заливкой), тень, номер кадра. Видео
1920×1080, 30 FPS, 1001 кадр, файлы Way1.FinalImage.CineCameraActor{N}.mp4 в --data, то есть ровно
в формате, который ждёт `stream.py replay`. Детекции берутся готовые (make_synthetic_data) — YOLO
по таким кадрам не запускается.

    python tools/render_synthetic_video.py --data tmp/synthetic --scenario base
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.cab_model import DEFAULT_DIMENSIONS, box_corners          # noqa: E402
from reconstruction.data_io import IMAGE_HEIGHT, IMAGE_WIDTH, video_path     # noqa: E402
from tools.make_synthetic_data import FPS, NUM_FRAMES, SCENARIOS, camera_poses, trajectory  # noqa: E402

# грани коробки по индексам углов _CORNERS (x: 0 перед/1 зад, y: ±, z: 0 низ/1 верх) -> индекс = 4x + 2y + z
FACES = {
    "front": ([0, 1, 3, 2], (70, 90, 200)), "back": ([4, 5, 7, 6], (60, 70, 150)),
    "left": ([0, 1, 5, 4], (90, 110, 220)), "right": ([2, 3, 7, 6], (80, 100, 210)),
    "top": ([1, 3, 7, 5], (120, 150, 240)), "bottom": ([0, 2, 6, 4], (40, 50, 120)),
}


def draw_frame(img, cam, corners, frame, grid):
    import cv2

    h, w = img.shape[:2]
    # небо и земля
    img[:] = (235, 225, 210)
    gp = cam.project(grid)
    ok = np.all(np.isfinite(gp), axis=1)
    # земля: заливка под линией горизонта по проекции сетки
    ys = gp[ok, 1]
    if ys.size:
        horizon = int(np.clip(ys.min(), 0, h))
        img[horizon:, :] = (120, 150, 125)
    n = int(np.sqrt(len(grid)))
    lines = gp.reshape(n, n, 2)
    for i in range(n):
        for pts in (lines[i], lines[:, i]):
            seg = pts[np.all(np.isfinite(pts), axis=1)]
            if len(seg) > 1:
                cv2.polylines(img, [seg.astype(np.int32)], False, (95, 120, 100), 1, cv2.LINE_AA)
    # тень на земле
    shadow = corners.copy()
    shadow[:, 2] = corners[:, 2].min()
    sp = cam.project(shadow)
    if np.all(np.isfinite(sp)):
        hull = cv2.convexHull(sp.astype(np.int32))
        cv2.fillPoly(img, [hull], (80, 100, 85))
    # грани коробки от дальних к ближним
    depth = cam.depth(corners)
    order = sorted(FACES.items(), key=lambda kv: -depth[kv[1][0]].mean())
    uv = cam.project(corners)
    if np.all(np.isfinite(uv)):
        for name, (idx, color) in order:
            poly = uv[idx].astype(np.int32)
            cv2.fillPoly(img, [poly], color)
            cv2.polylines(img, [poly], True, (30, 30, 60), 1, cv2.LINE_AA)
    cv2.putText(img, f"SYNTHETIC  frame {frame}", (w - 420, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2,
                cv2.LINE_AA)
    return img


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("tmp/synthetic"))
    p.add_argument("--scenario", default="base", choices=sorted(SCENARIOS))
    p.add_argument("--cameras", default="1,2,3,4,5")
    p.add_argument("--frames", type=int, default=NUM_FRAMES)
    args = p.parse_args(argv)
    import cv2  # noqa: F401

    sc = SCENARIOS[args.scenario]
    pos, direction = trajectory(np.arange(NUM_FRAMES, dtype=float), sc)
    cams = camera_poses(pos)
    corners_all = box_corners(pos, direction, DEFAULT_DIMENSIONS.as_array())
    lo, hi = pos.min(axis=0) - 6000, pos.max(axis=0) + 6000
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 25), np.linspace(lo[1], hi[1], 25))
    grid = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 150.0)]
    args.data.mkdir(parents=True, exist_ok=True)
    img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)
    for cam_id in (int(c) for c in args.cameras.split(",")):
        out = video_path(args.data, cam_id)
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                               "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
        for f in range(args.frames):
            draw_frame(img, cams[cam_id][f], corners_all[f], f, grid)
            ff.stdin.write(img.tobytes())
        ff.stdin.close()
        ff.wait()
        print(f"cam{cam_id}: {out}", flush=True)


if __name__ == "__main__":
    main()
