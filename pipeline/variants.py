"""
Сравнение вариантов архитектуры (ТЗ 6) на одних и тех же детекциях, в потоковом режиме detections.

  A  существующая realtime-система: рёбра рамки проекции модели, все камеры, без триангуляции и сплайна;
  B  A + классическая триангуляция центроидов масок как 3D-измерение фильтра;
  C  A + согласованная триангуляция (поправка ракурса по 3D-модели) + объединение в фильтре;
  D  C + скользящий B-сплайн (оценивается SMOOTHED TRACK с задержкой lag);
  E  D + адаптивный выбор камер (детекция не на всех камерах).

Для каждого варианта: MAE и RMSE по X/Y/Z, средняя/медианная/p95/максимальная 3D-ошибка, ошибка
скорости и ускорения, ошибка репроекции, число потерь трека и время повторного захвата, доля кадров
с оценкой, задержка и выдача координат в секунду, среднее число активных камер и детекций на тик.

Эталон — CSV цели, только для оценки. Опция --test-frames a:b[,c:d] ограничивает оценку отложенными
кадрами (остальные считаются «настроечными»); если разбиения нет, это указывается в отчёте.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from pipeline.center import _style, find_gaps, stats
from pipeline.config import StreamConfig
from pipeline.streaming import StreamPipeline, load_detections_table, load_truck_model
from reconstruction.bspline import fit_initial_bspline
from reconstruction.cab_model import truck_axes
from reconstruction.data_io import load_target

log = logging.getLogger("peleng.variants")

VARIANTS = {
    "A": ("Существующая система (до модернизации): рёбра рамки, курс = направление скорости, все камеры",
          {"scheduler.enabled": False, "triangulation.mode": "none", "spline.enabled": False,
           "tracker.heading_rate_limit_dps": 1e9}),
    "B": ("A + классическая триангуляция центроидов",
          {"scheduler.enabled": False, "triangulation.mode": "centroid", "spline.enabled": False}),
    "C": ("A + согласованная триангуляция + Калман",
          {"scheduler.enabled": False, "triangulation.mode": "model_corrected", "spline.enabled": False}),
    "D": ("C + скользящий B-сплайн (SMOOTHED)",
          {"scheduler.enabled": False, "triangulation.mode": "model_corrected", "spline.enabled": True}),
    "E": ("D + адаптивный выбор камер",
          {"scheduler.enabled": True, "triangulation.mode": "model_corrected", "spline.enabled": True}),
}
SMOOTHED_VARIANTS = {"D", "E"}


def variant_config(base: StreamConfig, key: str) -> StreamConfig:
    cfg = base.copy()
    cfg.streaming.mode = "detections"
    cfg.streaming.realtime_pacing = False
    cfg.output.video = False
    cfg.spline.background = False                   # детерминированное сравнение
    for dotted, value in VARIANTS[key][1].items():
        section, k = dotted.split(".")
        setattr(getattr(cfg, section), k, value)
    return cfg


def parse_ranges(text: str | None) -> list[tuple[int, int]]:
    if not text:
        return []
    out = []
    for part in text.split(","):
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def ground_truth(data_dir: Path, num_frames: int, fps: float, upm: float, frame_offset: int = 0) -> dict:
    target = load_target(data_dir)
    frames = np.arange(num_frames)
    gt = np.array([target.position(f + frame_offset) for f in frames])
    valid = np.all(np.isfinite(gt), axis=1)
    spline = fit_initial_bspline(target.frames[target.valid].astype(float), target.positions[target.valid], knot_spacing=5)
    t = np.clip((frames + frame_offset).astype(float), spline.t_start, spline.t_end)
    vel = spline.derivative(1).evaluate(t) * fps / upm
    acc = spline.derivative(2).evaluate(t) * fps ** 2 / upm
    return {"frames": frames, "gt": gt, "valid": valid, "speed": np.linalg.norm(vel, axis=1),
            "accel": np.linalg.norm(acc, axis=1), "issues": target.issues}


def reference_from_center(center: np.ndarray, heading_deg: float, model, dims: np.ndarray) -> np.ndarray:
    """Точка CSV по центру модели и курсу (та же геометрия, что в трекере)."""
    d = np.array([np.cos(np.radians(heading_deg)), np.sin(np.radians(heading_deg)), 0.0])
    forward, side, up = (a[0] for a in truck_axes(d[None, :]))
    if model is not None:
        return model.to_world(center, forward, side, up, model.csv_point[None, :])[0]
    return center + dims[0] / 2 * forward - dims[2] / 2 * up


def evaluate_records(records, truth: dict, upm: float, fps: float, model, dims: np.ndarray,
                     use_smoothed: bool = False, test_ranges: list[tuple[int, int]] | None = None) -> dict:
    n = len(truth["frames"])
    est = np.full((n, 3), np.nan)
    speed = np.full(n, np.nan)
    accel = np.full(n, np.nan)
    status = np.full(n, "none", dtype=object)
    cams_det = np.zeros(n, int)
    reproj = np.full(n, np.nan)
    std = np.full(n, np.nan)
    by_index = {r.index: r for r in records}
    for r in records:
        if r.index >= n:
            continue
        status[r.index] = r.status
        cams_det[r.index] = r.cameras_detected
        reproj[r.index] = r.reproj_px
        std[r.index] = r.position_std_m
        if use_smoothed:
            if np.isfinite(r.smoothed_index) and np.all(np.isfinite(r.smoothed_position)):
                i = int(round(r.smoothed_index))
                if 0 <= i < n:
                    src = by_index.get(i, r)
                    est[i] = reference_from_center(r.smoothed_position, src.heading_deg, model, dims)
                    speed[i] = np.linalg.norm(r.smoothed_velocity_mps)
                    accel[i] = r.smoothed_accel_mps2
        else:
            if np.all(np.isfinite(r.reference)):
                est[r.index] = r.reference
                speed[r.index] = r.speed_mps
                accel[r.index] = r.accel_mps2
    gt, valid = truth["gt"], truth["valid"]
    have = np.all(np.isfinite(est), axis=1)
    mask = have & valid
    if test_ranges:
        sel = np.zeros(n, bool)
        for a, b in test_ranges:
            sel[a:b + 1] = True
        mask &= sel
    diff = (est - gt) / upm
    err = np.linalg.norm(diff, axis=1)
    axes = {ax: {"mae": float(np.mean(np.abs(diff[mask, i]))), "rmse": float(np.sqrt(np.mean(diff[mask, i] ** 2))),
                 "bias": float(np.mean(diff[mask, i]))} if mask.any() else {} for i, ax in enumerate("xyz")}
    # потери трека: эпизоды predicted после инициализации; повторный захват — от первой детекции после
    # начала эпизода до первого measured
    initialized = np.array([s != "initializing" and s != "none" for s in status])
    predicted = np.array([s == "predicted" for s in status])
    episodes = [(a, b) for a, b in find_gaps(~predicted) if initialized[a]]
    reacq = []
    for a, b in episodes:
        first_det = next((f for f in range(a, n) if cams_det[f] > 0), None)
        first_meas = next((f for f in range(a, n) if status[f] == "measured"), None)
        if first_det is not None and first_meas is not None:
            reacq.append(first_meas - first_det)
    se = np.abs(speed - truth["speed"])
    ae = np.abs(accel - truth["accel"])
    out = {
        "frames_evaluated": int(mask.sum()),
        "availability": float(have.mean()),
        "error_3d_m": {"mean": float(err[mask].mean()) if mask.any() else float("nan"), **stats(err[mask])},
        "per_axis_m": axes,
        "speed_error_mps": stats(se[mask & np.isfinite(se)]),
        "accel_error_mps2": stats(ae[mask & np.isfinite(ae)]),
        "reprojection_px": stats(reproj[np.isfinite(reproj)]),
        "track_losses": len(episodes),
        "predicted_frames": int(predicted.sum()),
        "reacquisition_frames": {"mean": float(np.mean(reacq)) if reacq else 0.0, "max": int(max(reacq)) if reacq else 0,
                                 "episodes": len(reacq)},
        "position_std_median_m": float(np.nanmedian(std)) if np.any(np.isfinite(std)) else float("nan"),
        "error_series": err,
        "test_ranges": test_ranges or [],
    }
    return out


def run_variants(base: StreamConfig, data_dir: Path, det_dir: Path, out_dir: Path, keys: list[str],
                 num_frames: int | None = None, test_ranges=None, source_delays=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    detections, n = load_detections_table(det_dir, base.cameras.ids)
    n = num_frames or n
    c = base.cameras
    truth = ground_truth(data_dir, n, c.fps, c.units_per_meter, c.frame_offset)
    model = load_truck_model(base)
    results = {}
    for key in keys:
        cfg = variant_config(base, key)
        started = time.time()
        pipe = StreamPipeline(cfg, data_dir, out_dir / f"variant_{key}", detections, n, model=model,
                              source_delays=source_delays)
        records = pipe.run()
        dims = pipe.dims.as_array()
        m = evaluate_records(records, truth, c.units_per_meter, c.fps, model, dims, key in SMOOTHED_VARIANTS, test_ranges)
        if key in SMOOTHED_VARIANTS:
            m["realtime_track"] = {k: v for k, v in evaluate_records(records, truth, c.units_per_meter, c.fps, model,
                                                                     dims, False, test_ranges).items()
                                   if k != "error_series"}
        perf = pipe.performance()
        m["performance"] = {"updates_per_s": perf["updates_per_s"], "latency_ms": perf["latency_ms"],
                            "mean_active_cameras": perf["mean_active_cameras"],
                            "detections_per_tick": perf["detections_per_tick"],
                            "detection_fraction": perf["scheduler"].get("detection_fraction", 1.0),
                            "tick_ms_p50": perf["stages"]["tick_total"]["p50_ms"],
                            "tick_ms_p95": perf["stages"]["tick_total"]["p95_ms"]}
        m["name"] = VARIANTS[key][0]
        m["elapsed_s"] = time.time() - started
        results[key] = m
        e = m["error_3d_m"]
        log.info("вариант %s: медиана %.2f м, p95 %.2f, макс %.2f, кадров %d, активных камер %.2f (%.1f с)", key,
                 e.get("median", float("nan")), e.get("p95", float("nan")), e.get("max", float("nan")),
                 m["frames_evaluated"], m["performance"]["mean_active_cameras"], m["elapsed_s"])
    write_report(out_dir, results, truth, base, test_ranges)
    return results


def _f(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def write_report(out_dir: Path, results: dict, truth: dict, cfg: StreamConfig, test_ranges) -> None:
    lines = ["# Сравнение вариантов архитектуры", "",
             "> Эталон — CSV цели, только для оценки. YOLO (на реальных данных) обучена на этих же видео.",
             ""]
    if test_ranges:
        lines.append(f"Оценка только на отложенных кадрах: {', '.join(f'{a}–{b}' for a, b in test_ranges)}.")
    else:
        lines.append("Разбиения на настроечные и тестовые кадры нет: оценка по всем кадрам (не независимая оценка обобщения).")
    lines += ["", "## Точность (м, относительно CSV цели)", "",
              "| вариант | кадров | доля с оценкой | средняя | медиана | p95 | макс | MAE X / Y / Z | RMSE X / Y / Z |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, m in results.items():
        e, ax = m["error_3d_m"], m["per_axis_m"]
        lines.append(f"| {k}: {m['name']} | {m['frames_evaluated']} | {m['availability'] * 100:.0f} % | {_f(e.get('mean'))} | "
                     f"{_f(e.get('median'))} | {_f(e.get('p95'))} | {_f(e.get('max'))} | "
                     f"{_f(ax['x'].get('mae'))} / {_f(ax['y'].get('mae'))} / {_f(ax['z'].get('mae'))} | "
                     f"{_f(ax['x'].get('rmse'))} / {_f(ax['y'].get('rmse'))} / {_f(ax['z'].get('rmse'))} |")
    lines += ["", "## Кинематика, репроекция, потери", "",
              "| вариант | ошибка скорости, медиана / p95, м/с | ошибка ускорения, медиана / p95, м/с² | репроекция, медиана, px "
              "| потерь трека | кадров predicted | повторный захват, средн. / макс, кадров | СКО фильтра, медиана, м |",
              "|---|---|---|---|---|---|---|---|"]
    for k, m in results.items():
        se, ae, rp, rq = m["speed_error_mps"], m["accel_error_mps2"], m["reprojection_px"], m["reacquisition_frames"]
        lines.append(f"| {k} | {_f(se.get('median'))} / {_f(se.get('p95'))} | {_f(ae.get('median'))} / {_f(ae.get('p95'))} | "
                     f"{_f(rp.get('median'), 1)} | {m['track_losses']} | {m['predicted_frames']} | "
                     f"{_f(rq['mean'], 1)} / {rq['max']} | {_f(m['position_std_median_m'])} |")
    lines += ["", "## Производительность (режим detections: без стоимости YOLO)", "",
              "| вариант | тиков/с | задержка p50 / p95, мс | тик p50 / p95, мс | активных камер | детекций на тик | доля детекций |",
              "|---|---|---|---|---|---|---|"]
    for k, m in results.items():
        p = m["performance"]
        lines.append(f"| {k} | {_f(p['updates_per_s'], 1)} | {_f(p['latency_ms']['p50'], 1)} / {_f(p['latency_ms']['p95'], 1)} | "
                     f"{_f(p['tick_ms_p50'])} / {_f(p['tick_ms_p95'])} | {_f(p['mean_active_cameras'])} | "
                     f"{_f(p['detections_per_tick'])} | {_f(p['detection_fraction'])} |")
    for k, m in results.items():
        if "realtime_track" in m:
            r = m["realtime_track"]["error_3d_m"]
            lines.append(f"\nВариант {k}: REALTIME TRACK того же прогона — медиана {_f(r.get('median'))}, p95 {_f(r.get('p95'))}, "
                         f"макс {_f(r.get('max'))} м (SMOOTHED выше — с задержкой {cfg.spline.lag_frames} кадров).")
    lines += ["", "Графики: `variants/01_error_by_variant.png`, `variants/02_summary.png`.", ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "variants.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in m.items() if kk != "error_series"} for k, m in results.items()},
        ensure_ascii=False, indent=2, default=float))
    plot(out_dir, results, truth)


def plot(out_dir: Path, results: dict, truth: dict) -> None:
    plt = _style()
    colors = {"A": "#a3a29c", "B": "#eb6834", "C": "#2a78d6", "D": "#1baf7a", "E": "#7b5cd6"}
    frames = truth["frames"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for k, m in results.items():
        ax.plot(frames, m["error_series"], color=colors.get(k, "#000"), lw=1.6 if k in ("A", "E") else 1.2,
                label=f"{k}: {m['name']}")
    ax.set_ylim(0, None)
    ax.set_xlabel("Кадр")
    ax.set_ylabel("Ошибка, м")
    ax.set_title("Ошибка положения по кадрам по вариантам архитектуры")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "01_error_by_variant.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    keys = list(results)
    x = np.arange(len(keys))
    for ax, (title, getter) in zip(axes, (
            ("Ошибка положения, м", lambda m: (m["error_3d_m"].get("median", np.nan), m["error_3d_m"].get("p95", np.nan))),
            ("Ошибка скорости, м/с", lambda m: (m["speed_error_mps"].get("median", np.nan), m["speed_error_mps"].get("p95", np.nan))),
            ("Камер: активных / детекций на тик", lambda m: (m["performance"]["mean_active_cameras"], m["performance"]["detections_per_tick"])))):
        a = [getter(results[k])[0] for k in keys]
        b = [getter(results[k])[1] for k in keys]
        ax.bar(x - 0.18, a, 0.34, color="#2a78d6", label="медиана" if "Камер" not in title else "активных")
        ax.bar(x + 0.18, b, 0.34, color="#eda100", label="p95" if "Камер" not in title else "детекций")
        for xi, v in zip(np.r_[x - 0.18, x + 0.18], a + b):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x, keys)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_summary.png")
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--data", type=Path, default=Path("tmp/project2"))
    p.add_argument("--detections", type=Path, default=Path("outputs/main/detections"))
    p.add_argument("--out", type=Path, default=Path("outputs/main/variants"))
    p.add_argument("--variants", default="A,B,C,D,E")
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--test-frames", default=None, help="отложенные кадры для оценки, например 300:600,800:1000")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = StreamConfig.load(args.config, args.set)
    results = run_variants(base, args.data, args.detections, args.out, args.variants.split(","), args.num_frames,
                           parse_ranges(args.test_frames))
    print((args.out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
