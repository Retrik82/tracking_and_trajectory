"""
Ошибка потокового трека относительно CSV цели (только для оценки, не для работы системы).

    python tools/eval_stream_track.py --track outputs/stream_tuned_rt/stream_track.csv --data D:\\project2

Берёт колонку ref_* (точка CSV на модели) из stream_track.csv и сравнивает с положением цели
на том же кадре. Считает только тики со статусом measured/partial (predicted — прогноз, его
ошибка показательна отдельно: --include-predicted).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reconstruction.data_io import load_target  # noqa: E402


def evaluate(track_csv: Path, data_dir: Path, units_per_meter: float = 100.0,
             include_predicted: bool = False) -> dict:
    target = load_target(data_dir)
    statuses = ("measured", "partial", "predicted") if include_predicted else ("measured", "partial")
    errors, frames = [], []
    with open(track_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] not in statuses:
                continue
            ref = np.array([float(row["ref_x"]), float(row["ref_y"]), float(row["ref_z"])])
            truth = target.position(int(row["frame"]))
            if truth is None or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(ref)):
                continue
            errors.append(float(np.linalg.norm(ref - truth)) / units_per_meter)
            frames.append(int(row["frame"]))
    e = np.array(errors)
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "median_m": float(np.median(e)), "p95_m": float(np.percentile(e, 95)),
            "max_m": float(e.max()), "first_frame": frames[0], "last_frame": frames[-1]}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--units-per-meter", type=float, default=100.0)
    p.add_argument("--include-predicted", action="store_true")
    args = p.parse_args(argv)
    r = evaluate(args.track, args.data, args.units_per_meter, args.include_predicted)
    if not r["n"]:
        print("нет тиков с оценкой")
        return
    print(f"тиков с оценкой: {r['n']} (кадры {r['first_frame']}–{r['last_frame']}); "
          f"ошибка точки CSV: медиана {r['median_m']:.2f}, p95 {r['p95_m']:.2f}, макс {r['max_m']:.2f} м")


if __name__ == "__main__":
    main()
