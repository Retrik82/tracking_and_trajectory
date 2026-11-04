"""
Метрики «дёрганья» трека без эталона (по stream_track.csv потокового конвейера).

  ускорение по траектории (вторая разность центра кабины), м/с²  — скачки положения;
  скорость поворота курса, °/с                                     — дёрганье корпуса;
  скачки: кадры, где ускорение > 20 м/с² (физически невозможно для грузовика);
  отклонение от офлайн-сглаженного трека (центрированное окно 1 с) — ТОЛЬКО для оценки «шум vs. запаздывание»:
  трекер сглаживать будущими кадрами нельзя, а сравнение показывает, сколько в треке высокочастотного шума.

python tools/jitter_metrics.py outputs/scene2_all/stream/stream_track.csv [ещё csv ...]
"""
import csv
import sys

import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    f = np.array([int(r["frame"]) for r in rows])
    pre = sys.argv[1] if sys.argv[1] in ("cab_", "", "s") else "cab_"
    p = np.array([[float(r[pre + "x"]), float(r[pre + "y"]), float(r[pre + "z"])] for r in rows]) / 100.0
    h = np.array([float(r["heading_deg"]) for r in rows])
    st = np.array([r["status"] for r in rows])
    return f, p, h, st


def metrics(path, fps=30.0):
    f, p, h, st = load(path)
    ok = np.all(np.isfinite(p), axis=1)
    acc = np.full(len(p), np.nan)
    acc[1:-1] = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) * fps ** 2
    dh = np.full(len(h), np.nan)
    dh[1:] = np.abs((np.diff(h) + 180) % 360 - 180) * fps
    # офлайн-эталон гладкости: центрированное скользящее среднее 31 кадр (только для оценки)
    k = 31
    smooth = np.full_like(p, np.nan)
    for i in range(k // 2, len(p) - k // 2):
        w = p[i - k // 2:i + k // 2 + 1]
        if np.all(np.isfinite(w)):
            smooth[i] = w.mean(axis=0)
    dev = np.linalg.norm(p - smooth, axis=1)
    a, d, e = acc[np.isfinite(acc)], dh[np.isfinite(dh)], dev[np.isfinite(dev)]
    return {
        "кадров с координатами": int(ok.sum()),
        "ускорение p50/p95/max, м/с²": f"{np.median(a):.1f} / {np.percentile(a, 95):.1f} / {a.max():.0f}",
        "скачков (>20 м/с²)": int((a > 20).sum()),
        "поворот курса p95/max, °/с": f"{np.percentile(d, 95):.0f} / {d.max():.0f}",
        "поворотов >90 °/с": int((d > 90).sum()),
        "шум относительно сглаженного p50/p95, м": f"{np.median(e):.2f} / {np.percentile(e, 95):.2f}",
    }


if __name__ == "__main__":
    for path in [a for a in sys.argv[1:] if a.endswith(".csv")]:
        print(path)
        for k, v in metrics(path).items():
            print(f"  {k}: {v}")
