"""
Пайплайн «Пеленг»: траектория грузовика по синхронным видео пяти камер.

Шаги (по порядку; каждый сохраняет результат в <out>, следующий можно запускать отдельно):
  detect    YOLO по всем видео -> <out>/detections (кешируется)                 pipeline/detect.py
  center    офлайн: центр маски -> триангуляция -> начальный B-сплайн -> <out>/center
                                                                                  pipeline/center.py
  optimize  офлайн: сплайн центра маски, подогнанный по пикселям всех камер     reconstruction/optimizer.py
  cab       офлайн: модель коробки, подогнанная ко всем рамкам всего видео      reconstruction/cab_model.py
  realtime  ОСНОВНОЙ: трекер реального времени (только прошлые и текущий кадры)
            с 3D-моделью грузовика и обработкой потери видимости             reconstruction/visibility.py
  compare   все варианты против CSV цели: <out>/report.md, metrics.json, compare/*.png
  video     видео с силуэтом модели и точкой кабины трекера: <out>/videos (флаг --video)

center, optimize и cab обрабатывают весь видеоряд сразу (используют будущие кадры) и нужны
только для сравнения. center при этом нужен всегда: его сплайн — начальное приближение для cab.
"""

import argparse
import copy
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from pipeline.center import (
    AQUA, BLUE, CAMERA_COLORS, CAMERAS, INK_2, MAGENTA, MUTED, NUM_VIDEO_FRAMES, ORANGE, YELLOW,
    _style, debiased_norm, find_gaps, run_evaluation, stats, truck_frame_components,
)
from pipeline.coverage import plot_camera_coverage
from pipeline.detect import detector_info, load_detections, run_detection
from reconstruction.bspline import BSplineTrajectory, fit_initial_bspline
from reconstruction.cab_model import (
    CAB_CENTER_BACK,
    CAB_CENTER_UP,
    DEFAULT_DIMENSIONS,
    BoxObservations,
    TruckDimensions,
    box_corners,
    fit_cab_trajectory,
    initial_cab_trajectory,
    offset_point_trajectory,
    truck_axes,
)
from reconstruction.data_io import IMAGE_HEIGHT, IMAGE_WIDTH, load_target, make_camera, video_path
from reconstruction.objective import ObjectiveData
from reconstruction.optimizer import optimize_trajectory
from reconstruction.truck_model import SCENE_SCALE, TruckModel
from reconstruction.visibility import INITIALIZING, MEASURED, PARTIAL, PREDICTED, TrackerSettings, run_tracker

log = logging.getLogger("peleng")
ALL_STEPS = ["detect", "center", "optimize", "cab", "realtime", "compare"]
MAIN_METHOD = "realtime"
METHOD_NAMES = {
    "triangulation": "Офлайн. Центр маски: триангуляция",
    "center": "Офлайн. Центр маски: начальный сплайн",
    "optimized": "Офлайн. Центр маски: оптимизация по пикселям",
    "cab": "Офлайн. Кабина: низ передней грани",
    "cab_center": "Офлайн. Кабина: центр кабины",
    "realtime_ref": "Реальное время: низ передней грани",      # с 3D-моделью — «точка CSV цели на модели»
    "realtime": "Реальное время: центр кабины (основной)",
}
METHOD_COLORS = {"triangulation": ORANGE, "center": MUTED, "optimized": MAGENTA, "cab": INK_2, "cab_center": AQUA,
                 "realtime_ref": YELLOW, "realtime": BLUE}


def build_main_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"), help="веса YOLO (.pt)")
    p.add_argument("--data", type=Path, default=Path("tmp/project2"), help="папка с видео и CSV")
    p.add_argument("--out", type=Path, default=Path("outputs/main"))
    p.add_argument("--steps", default=",".join(ALL_STEPS), help=f"через запятую из: {', '.join(ALL_STEPS)}")
    p.add_argument("--video", action="store_true", help="добавить шаг video")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--units-per-meter", type=float, default=100.0, help="единиц мира в метре (Unreal: см)")
    p.add_argument("--frame-offset", type=int, default=0, help="кадр CSV = кадр видео + смещение")

    g = p.add_argument_group("detect")
    g.add_argument("--redetect", action="store_true", help="пересчитать детекции, даже если они сохранены")
    g.add_argument("--detector", choices=["yolo", "classic"], default="yolo",
                   help="yolo — нейросеть; classic — вычитание фона без нейросети (detection/classic.py)")
    g.add_argument("--imgsz", type=int, default=960)
    g.add_argument("--conf", type=float, default=0.25)
    g.add_argument("--device", default=None, help="mps / cuda / cpu; по умолчанию — автоматически")
    g.add_argument("--min-mask-area", type=float, default=400.0, help="маска меньше — флаг too_small, px")
    g.add_argument("--min-reliable-conf", type=float, default=0.5)

    g = p.add_argument_group("center (офлайн)")
    g.add_argument("--max-reproj-px", type=float, default=20.0, help="порог отбрасывания камеры в триангуляции")
    g.add_argument("--knot-spacing", type=float, default=15.0, help="шаг узлов сплайна, кадры")
    g.add_argument("--huber-scale", type=float, default=100.0, help="порог Huber для сплайна, единицы мира")

    g = p.add_argument_group("optimize и cab (офлайн)")
    g.add_argument("--cab-knot-spacing", type=float, default=10.0, help="шаг узлов сплайна кабины, кадры")
    g.add_argument("--fixed-dimensions", action="store_true", help="не подбирать габариты коробки")
    g.add_argument("--lambda-acceleration", type=float, default=10.0)
    g.add_argument("--lambda-jerk", type=float, default=100.0)
    g.add_argument("--max-nfev", type=int, default=3000)

    g = p.add_argument_group("грузовик")
    g.add_argument("--truck-fbx", type=Path, default=Path("models/CargoTruck.fbx"),
                   help="3D-модель для трекера реального времени; если файла нет — коробка")
    g.add_argument("--truck-scale", type=float, default=SCENE_SCALE, help="масштаб актора грузовика в сцене")
    g.add_argument("--truck-length", type=float, default=DEFAULT_DIMENSIONS.length, help="коробка: длина, ед. мира")
    g.add_argument("--truck-width", type=float, default=DEFAULT_DIMENSIONS.width)
    g.add_argument("--truck-height", type=float, default=DEFAULT_DIMENSIONS.height)
    g.add_argument("--cab-center-back", type=float, default=CAB_CENTER_BACK,
                   help="коробка: центр кабины на столько назад от передней грани, ед. мира")
    g.add_argument("--cab-center-up", type=float, default=CAB_CENTER_UP,
                   help="коробка: центр кабины на столько выше низа передней грани, ед. мира")
    return p


def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", encoding="utf-8")],
    )


def camera_provider(args):
    return lambda cam, frame: make_camera(args.data, cam, frame)


# ---------------------------------------------------------------- шаги

def step_center(args, detections) -> dict:
    center_args = copy.copy(args)
    center_args.out = args.out / "center"
    center_args.detector_info = detector_info(args.out)
    result = run_evaluation(center_args, detections)
    np.savez(args.out / "center_spline.npz", knots=result["spline"].knots,
             control_points=result["spline"].control_points)
    return result


def load_spline(path: Path) -> BSplineTrajectory:
    d = np.load(path)
    return BSplineTrajectory(d["knots"], d["control_points"], 3, float(d["knots"][0]), float(d["knots"][-1]))


def step_optimize(args, detections, center: BSplineTrajectory) -> BSplineTrajectory:
    rows = [d for cam in detections.values() for d in cam.values()
            if not d.unreliable and center.t_start <= d.frame <= center.t_end]
    data = ObjectiveData(
        times=np.array([d.frame for d in rows], float),
        camera_indices=np.array([d.camera for d in rows]),
        pixels=np.array([[d.point_u, d.point_v] for d in rows]),
        confidence=np.array([d.confidence for d in rows]),
        frames=np.array([d.frame + args.frame_offset for d in rows]),
    )
    started = time.time()
    result = optimize_trajectory(center, camera_provider(args), data, max_nfev=2000,
                                 lambda_acceleration=args.lambda_acceleration, lambda_jerk=args.lambda_jerk)
    log.info("оптимизация центра маски: %d наблюдений, %.1f с, %s", len(rows), time.time() - started,
             result.scipy_result.message)
    np.savez(args.out / "optimized_spline.npz", knots=result.trajectory.knots,
             control_points=result.trajectory.control_points)
    return result.trajectory


def step_cab(args, detections, center: BSplineTrajectory) -> dict:
    observations = BoxObservations.from_detections(
        [d for cam in detections.values() for d in cam.values()],
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT, frame_offset=args.frame_offset,
    )
    dims = TruckDimensions(args.truck_length, args.truck_width, args.truck_height)
    initial = initial_cab_trajectory(center, dims, knot_spacing=args.cab_knot_spacing)
    started = time.time()
    fit = fit_cab_trajectory(
        initial, camera_provider(args), observations, dimensions=dims,
        fit_dimensions=not args.fixed_dimensions, lambda_acceleration=args.lambda_acceleration,
        lambda_jerk=args.lambda_jerk, max_nfev=args.max_nfev,
    )
    d = fit.dimensions
    log.info("кабина: %d рамок, %.1f с, %s; габариты %.2f × %.2f × %.2f м", len(observations.times),
             time.time() - started, fit.scipy_result.message, d.length / args.units_per_meter,
             d.width / args.units_per_meter, d.height / args.units_per_meter)
    np.savez(args.out / "cab_spline.npz", knots=fit.trajectory.knots, control_points=fit.trajectory.control_points,
             dimensions=d.as_array())
    center_trajectory = offset_point_trajectory(fit.trajectory, args.cab_center_back, args.cab_center_up)
    np.savez(args.out / "cab_center_spline.npz", knots=center_trajectory.knots,
             control_points=center_trajectory.control_points,
             offset_back_up=np.array([args.cab_center_back, args.cab_center_up]))
    return {"trajectory": fit.trajectory, "center_trajectory": center_trajectory, "dimensions": d,
            "edge_errors_px": fit.edge_errors_px,
            "converged": bool(fit.scipy_result.success), "nfev": int(fit.scipy_result.nfev),
            "observations": observations}


def step_realtime(args, detections) -> dict:
    """Трекер реального времени: кадры по порядку, на каждом — только прошлое и текущий кадр."""
    rows = [d for cam in detections.values() for d in cam.values()]
    observations = BoxObservations.from_detections(rows, width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                                                    frame_offset=args.frame_offset)
    points = {(d.camera, d.frame): (d.point_u, d.point_v) for d in rows}
    reliable = {(d.camera, d.frame): not d.unreliable for d in rows}
    settings = TrackerSettings(fps=args.fps, units_per_meter=args.units_per_meter,
                               cab_center_back=args.cab_center_back, cab_center_up=args.cab_center_up)
    dims = TruckDimensions(args.truck_length, args.truck_width, args.truck_height)
    model = None
    if args.truck_fbx and Path(args.truck_fbx).exists():
        model = TruckModel.from_fbx(args.truck_fbx, scale=args.truck_scale)
        log.info("трекер: 3D-модель %s, масштаб %.3f, габариты %s м", args.truck_fbx, args.truck_scale,
                 " × ".join(f"{x:.2f}" for x in model.dimensions / args.units_per_meter))
        METHOD_NAMES["realtime_ref"] = "Реальное время: точка CSV цели на модели"
    else:
        log.warning("трекер: 3D-модели %s нет, используется коробка %s", args.truck_fbx, dims)
    cache: dict = {}

    def provider(cam, frame):
        if (cam, frame) not in cache:
            cache[(cam, frame)] = make_camera(args.data, cam, frame)
        return cache[(cam, frame)]

    started = time.time()
    result = run_tracker(observations, provider, np.arange(NUM_VIDEO_FRAMES), points, reliable, settings, dims, model)
    elapsed = time.time() - started
    status = result.array("status")
    log.info("реальное время: %.1f мс/кадр (с чтением поз камер); статусы %s", elapsed / NUM_VIDEO_FRAMES * 1000,
             {k: int((status == k).sum()) for k in (INITIALIZING, MEASURED, PARTIAL, PREDICTED)})
    np.savez(args.out / "realtime_track.npz", frames=np.arange(NUM_VIDEO_FRAMES), status=status,
             reference=result.array("position"), box_center=result.array("box_center"),
             cab_center=result.array("cab_center"), speed_mps=result.array("speed_mps"),
             heading=result.array("heading"), position_std=result.array("position_std"),
             cameras=result.array("cameras"), cameras_used=result.array("cameras_used"),
             occluded_edges=result.array("occluded_edges"),
             frames_since_measurement=result.array("frames_since_measurement"))
    return {"result": result, "dimensions": dims, "model": model, "ms_per_frame": elapsed / NUM_VIDEO_FRAMES * 1000}


# ---------------------------------------------------------------- сравнение

def evaluate_methods(args, detections, center_result, optimized, cab, realtime) -> tuple[dict, dict]:
    upm, fps = args.units_per_meter, args.fps
    target = load_target(args.data)
    frames = np.arange(NUM_VIDEO_FRAMES)
    gt = np.array([target.position(f + args.frame_offset) for f in frames])
    gt_valid = np.all(np.isfinite(gt), axis=1)
    gt_spline = fit_initial_bspline(target.frames[target.valid].astype(float), target.positions[target.valid],
                                    knot_spacing=5)
    csv_t = np.clip((frames + args.frame_offset).astype(float), gt_spline.t_start, gt_spline.t_end)
    gt_vel = gt_spline.derivative().evaluate(csv_t)
    gt_speed = np.linalg.norm(gt_vel, axis=1) * fps / upm
    moving = np.linalg.norm(gt_vel, axis=1) * fps / upm > 1.0

    ctx = center_result["ctx"]
    detected_cams = np.array([sum(f in detections[c] for c in CAMERAS) for f in frames])
    series = {"triangulation": {"points": np.where(ctx["ok"][:, None], ctx["tri_points"], np.nan), "spline": None}}
    for name, spline in (("center", center_result["spline"]), ("optimized", optimized), ("cab", cab["trajectory"]),
                         ("cab_center", cab["center_trajectory"])):
        series[name] = {"points": np.full((NUM_VIDEO_FRAMES, 3), np.nan), "spline": spline}
        span = (frames >= spline.t_start) & (frames <= spline.t_end)
        series[name]["points"][span] = spline.evaluate(frames[span].astype(float))
    track = realtime["result"]
    rt_speed = track.array("speed_mps")
    series["realtime_ref"] = {"points": track.array("position"), "spline": None, "speed": rt_speed}
    series["realtime"] = {"points": track.array("cab_center"), "spline": None, "speed": rt_speed}

    metrics = {}
    for name, s in series.items():
        pts = s["points"]
        has = np.all(np.isfinite(pts), axis=1)
        err = np.linalg.norm(pts - gt, axis=1) / upm
        along, cross, vertical = truck_frame_components((pts - gt) / upm, gt_vel)
        comp_mask = has & gt_valid & moving
        debiased, bias = debiased_norm(along, cross, vertical, comp_mask)
        m = {
            "frames": int((has & gt_valid).sum()),
            "position_m": stats(err[has & gt_valid]),
            "bias_m": bias,
            "position_without_bias_m": stats(debiased[comp_mask]),
            "by_detected_cameras": {int(k): stats(err[has & gt_valid & (detected_cams == k)])
                                    for k in range(1, 5) if np.any(has & gt_valid & (detected_cams == k))},
        }
        if s["spline"] is not None or "speed" in s:
            span = has
            if s["spline"] is not None:
                vel = np.full((NUM_VIDEO_FRAMES, 3), np.nan)
                vel[span] = s["spline"].derivative(1).evaluate(frames[span].astype(float)) * fps / upm
                s["speed"] = np.linalg.norm(vel, axis=1)
            speed = s["speed"]
            # Одинаковая для всех вариантов мера скачков: изменение скорости между кадрами, м/с²
            jumps = np.abs(np.diff(speed)) * fps
            jumps = jumps[np.isfinite(jumps)]
            path = float(np.nansum(np.linalg.norm(np.diff(pts[span], axis=0), axis=1)) / upm)
            gt_path = float(np.sum(np.linalg.norm(np.diff(gt_spline.evaluate(csv_t[span]), axis=0), axis=1)) / upm)
            m["kinematics"] = {
                "speed_max_mps": float(np.nanmax(speed)),
                "speed_error_mps": stats(np.abs(speed - gt_speed)[span]),
                "speed_change_max_mps2": float(jumps.max()),
                "speed_change_p99_mps2": float(np.percentile(jumps, 99)),
                "path_m": path, "target_path_m": gt_path,
            }
        s["error"] = err
        metrics[name] = m

    target_jumps = np.abs(np.diff(gt_speed)) * fps
    ctx_out = dict(frames=frames, gt=gt, gt_valid=gt_valid, gt_speed=gt_speed, series=series,
                   detected_cams=detected_cams, target_issues=target.issues,
                   target_speed_change_p99=float(np.percentile(target_jumps, 99)), realtime=realtime)
    return metrics, ctx_out


def plot_comparison(out: Path, c: dict, cab: dict, args) -> None:
    plt = _style()
    upm, frames = args.units_per_meter, c["frames"]
    note = detector_info(args.out)["note"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in ("center", "optimized", "cab_center", "realtime"):
        ax.plot(frames, c["series"][name]["error"], color=METHOD_COLORS[name], lw=2.5 if name == MAIN_METHOD else 1.5,
                label=METHOD_NAMES[name])
    ax.scatter(frames, c["series"]["triangulation"]["error"], s=5, color=ORANGE, alpha=0.5, lw=0,
               label=METHOD_NAMES["triangulation"])
    ax.set_ylabel("Ошибка относительно CSV цели, м")
    ax.set_xlabel("Кадр видео")
    ax.set_ylim(0, None)
    ax.set_title(f"Ошибка положения по кадрам — {note}")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "01_error_by_method.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    gt = c["gt"] / upm
    ax.plot(gt[:, 0], gt[:, 1], color=INK_2, lw=2, label="цель (CSV)")
    for name in ("cab_center", "realtime"):
        p = c["series"][name]["points"] / upm
        ax.plot(p[:, 0], p[:, 1], color=METHOD_COLORS[name], lw=2, label=METHOD_NAMES[name])
    ax.set_aspect("equal")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.set_title("Траектория, вид сверху")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "02_trajectory_top.png")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(frames, c["gt_speed"], color=INK_2, lw=2, label="цель (CSV)")
    for name in ("cab_center", "realtime"):
        a1.plot(frames, c["series"][name]["speed"], color=METHOD_COLORS[name], lw=1.8, label=METHOD_NAMES[name])
        a2.plot(frames, c["series"][name]["speed"] - c["gt_speed"], color=METHOD_COLORS[name], lw=1.5)
    a2.axhline(0, color=MUTED, lw=1)
    a1.set_ylabel("Скорость, м/с")
    a2.set_ylabel("Разница, м/с")
    a2.set_xlabel("Кадр видео")
    a1.set_title("Скорость из производной сплайна")
    a1.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "03_speed.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4.6))
    names = list(METHOD_NAMES)
    x = np.arange(len(names))
    med = [c["metrics"][n]["position_m"]["median"] for n in names]
    p95 = [c["metrics"][n]["position_m"]["p95"] for n in names]
    ax.bar(x - 0.18, med, width=0.34, color=BLUE, label="медиана")
    ax.bar(x + 0.18, p95, width=0.34, color=YELLOW, label="p95")
    for xi, v in zip(x - 0.18, med):
        ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    for xi, v in zip(x + 0.18, p95):
        ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x, [METHOD_NAMES[n].replace(". ", ".\n").replace(": ", ":\n") for n in names], fontsize=8)
    ax.set_ylabel("Ошибка, м")
    ax.set_title("Сравнение вариантов относительно CSV цели")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "04_methods_summary.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    obs = cab["observations"]
    cam_idx = obs.camera_indices
    if len(cab["edge_errors_px"]) != len(cam_idx):
        # После подгонки ошибки рёбер есть только для рамок внутри интервала сплайна
        cam_idx = cam_idx[(obs.times >= cab["trajectory"].t_start) & (obs.times <= cab["trajectory"].t_end)]
    data, labels = [], []
    for cam in CAMERAS:
        e = cab["edge_errors_px"][cam_idx == cam]
        e = e[np.isfinite(e)]
        data.append(e)
        labels.append(f"Cam{cam}\nn={len(e)}")
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.5)
    for patch, cam in zip(bp["boxes"], CAMERAS):
        patch.set_facecolor(CAMERA_COLORS[cam])
        patch.set_alpha(0.8)
    ax.set_ylabel("Пиксели")
    ax.set_title("Рёбра рамки коробки против рамки детектора")
    fig.tight_layout()
    fig.savefig(out / "05_cab_edge_errors.png")
    plt.close(fig)


def _fmt(s: dict) -> str:
    if not s or not s.get("n"):
        return "—"
    return f"{s['median']:.2f} / {s['p95']:.2f} / {s['rmse']:.2f} / {s['max']:.2f}"


def write_report(args, metrics: dict, cab: dict, c: dict) -> None:
    d = cab["dimensions"]
    upm = args.units_per_meter
    lines = [
        "# Траектория грузовика: точка кабины",
        "",
        "> ⚠️ Детектор YOLO дообучен на этих же видео: качество детекции оценено на обучающих данных.",
        "",
        "> Основной результат считается **в реальном времени**: оценка на кадре t использует только кадры ≤ t. "
        "Варианты с пометкой «Офлайн» обрабатывают весь видеоряд сразу (заглядывают вперёд) и приведены "
        "только для сравнения.",
        "",
        f"Основной результат — **{METHOD_NAMES[MAIN_METHOD]}**. Детекции: {detector_info(args.out)['source']}, "
        f"смещение кадров: {args.frame_offset}.",
        "",
        "Испорченные ячейки CSV цели (исключены из оценки): " + ("; ".join(c["target_issues"]) or "нет"),
        "",
        f"Габариты коробки: {d.length / upm:.2f} × {d.width / upm:.2f} × {d.height / upm:.2f} м "
        f"({'фиксированы' if args.fixed_dimensions else 'подобраны по рамкам'}; "
        "это «эффективная» коробка для рамок детектора, а не реальный размер грузовика). "
        f"Оптимизация: {'сошлась' if cab['converged'] else 'не сошлась'}, итераций {cab['nfev']}.",
        "",
        f"Центр кабины — точка на {args.cab_center_back / upm:.2f} м назад от передней грани и на "
        f"{args.cab_center_up / upm:.2f} м выше её низа (допущение о типичной кабине, задаётся "
        "`--cab-center-back`/`--cab-center-up`). Точка CSV цели — опорная точка модели в Unreal, она ниже "
        "и впереди центра кабины, поэтому ошибка центра кабины относительно CSV включает это постоянное "
        "смещение. Строка «низ передней грани» показывает точность той же подгонки в точке, ближайшей к CSV.",
        "",
        "## Сравнение с CSV цели, м (медиана / p95 / RMSE / максимум)",
        "",
        "| Вариант | кадров | ошибка положения | без постоянного смещения | смещение вдоль / поперёк / верт., м "
        "| макс. скорость, м/с | ошибка скорости, медиана / p95, м/с | скачок скорости между кадрами, p99 / макс, м/с² "
        "| путь, м (цель) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        b = m["bias_m"]
        k = m.get("kinematics")
        kin = (f"{k['speed_max_mps']:.1f} | {k['speed_error_mps']['median']:.2f} / {k['speed_error_mps']['p95']:.2f} "
               f"| {k['speed_change_p99_mps2']:.1f} / {k['speed_change_max_mps2']:.1f} "
               f"| {k['path_m']:.0f} ({k['target_path_m']:.0f})") if k else "— | — | — | —"
        title = f"**{METHOD_NAMES[name]}**" if name == MAIN_METHOD else METHOD_NAMES[name]
        lines.append(f"| {title} | {m['frames']} | {_fmt(m['position_m'])} | {_fmt(m['position_without_bias_m'])} "
                     f"| {b['along_m']:+.2f} / {b['cross_m']:+.2f} / {b['vertical_m']:+.2f} | {kin} |")
    rt = c["realtime"]["result"]
    status = rt.array("status")
    since = rt.array("frames_since_measurement")
    lines += ["", f"Скорость цели по CSV: максимум ≈ 32 м/с; скачок скорости между кадрами p99 "
              f"{c['target_speed_change_p99']:.1f} м/с².", "",
              "## Реальное время: статусы кадров", "",
              f"Время на кадр: {c['realtime']['ms_per_frame']:.1f} мс (вместе с чтением поз камер).", "",
              "| статус | кадров | что значит |", "|---|---|---|",
              f"| initializing | {(status == INITIALIZING).sum()} | ещё нет оценки (нужно несколько кадров с ≥2 камерами) |",
              f"| measured | {(status == MEASURED).sum()} | хотя бы одна камера дала ≥3 ребра рамки |",
              f"| partial | {(status == PARTIAL).sum()} | детекции есть, но рёбра закрыты препятствием или краем кадра |",
              f"| predicted | {(status == PREDICTED).sum()} | детекций нет, прогноз по вектору движения |",
              "",
              f"Рёбер рамок, признанных закрытыми препятствием: {int(rt.array('occluded_edges').sum())}. "
              f"Максимум кадров подряд без измерения: {int(since.max())}.", "",
              "## Основной результат по числу камер с детекцией на кадре, м", "",
              "| камер | кадров | медиана | p95 | максимум |", "|---|---|---|---|---|"]
    for k, s in metrics[MAIN_METHOD]["by_detected_cameras"].items():
        lines.append(f"| {k} | {s['n']} | {s['median']:.2f} | {s['p95']:.2f} | {s['max']:.2f} |")
    gaps = [(a, b) for a, b in find_gaps(c["detected_cams"] > 0) if b - a >= 2]
    lines += ["", "Кадры без единой детекции (≥3 подряд): " + (", ".join(f"{a}–{b}" for a, b in gaps) or "нет"), "",
              "## Графики", ""]
    lines += [f"![{p.stem}](compare/{p.name})" for p in sorted((args.out / "compare").glob("*.png"))]
    lines += ["", "Отчёт базового варианта (центр маски): [center/report.md](center/report.md)", "",
              "## Известные ограничения реального времени", "",
              "- Курс берётся из направления скорости фильтра. В медленном резком повороте (кадры ~650–850, "
              "~5–6 м/с) он ошибается на десятки градусов, и точка кабины уходит вбок примерно на половину длины.",
              "- Участок, где грузовик видит одна камера (кадры ~200–300), даёт большую ошибку по глубине.",
              "- С 3D-моделью: масштаб актора ×3.186 подтверждается и без CSV цели (лучшее совпадение с рамками YOLO "
              "при ×3.0–3.19), а положение точки CSV на модели (передний край кабины, 3 м над дорогой) "
              "откалибровано по CSV цели — сверьте с актором в Unreal.",
              "- Без 3D-модели — коробка 16.5 × 2.55 × 4.0 м (онлайн-уточнение габаритов расходится).",
              "- Пределы ускорения и СКО ребра рамки подобраны под эту сцену (по CSV цели)."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- видео

_BOX_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def render_videos(args, detections, c: dict) -> None:
    video_note = detector_info(args.out)["note_en"]
    """Видео по трекеру реального времени: на кадре t рисуется только то, что известно к кадру t."""
    import cv2

    out_dir = args.out / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    track = c["realtime"]["result"]
    dims = c["realtime"]["dimensions"].as_array()
    status = track.array("status")
    ref = track.array("position")
    pts = track.array("cab_center")
    heading = track.array("heading")
    span = np.all(np.isfinite(ref), axis=1)
    direction = np.c_[np.cos(heading), np.sin(heading), np.zeros(len(heading))]
    model = c["realtime"]["model"]
    centers = track.array("box_center")
    corners = np.full((NUM_VIDEO_FRAMES, 8, 3), np.nan)
    if model is None:
        corners[span] = box_corners(ref[span], direction[span], dims)
    speed = c["series"][MAIN_METHOD]["speed"]
    err = c["series"][MAIN_METHOD]["error"]
    upm = args.units_per_meter

    def bgr(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))

    for cam in CAMERAS:
        cap = cv2.VideoCapture(str(video_path(args.data, cam)))
        target_file = out_dir / f"realtime_cam{cam}.mp4"
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
             "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-pix_fmt", "yuv420p", str(target_file)], stdin=subprocess.PIPE)
        for f in range(NUM_VIDEO_FRAMES):
            ok, img = cap.read()
            if not ok:
                break
            camera = make_camera(args.data, cam, f + args.frame_offset)
            d = detections[cam].get(f)
            if d is not None:
                cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), bgr(MUTED), 1, cv2.LINE_AA)
            if span[f] and model is not None:
                fw, sd, up = (a[0] for a in truck_axes(direction[f][None, :]))
                hull = camera.project(model.to_world(centers[f], fw, sd, up, model.hull))
                hull = hull[np.all(np.isfinite(hull), axis=1)].astype(np.int32)
                if len(hull) > 2:
                    cv2.polylines(img, [cv2.convexHull(hull)], True, bgr(BLUE), 2, cv2.LINE_AA)
            elif span[f]:
                box = camera.project(corners[f])
                for a, b in _BOX_EDGES:
                    if np.all(np.isfinite(box[[a, b]])):
                        cv2.line(img, tuple(box[a].astype(int)), tuple(box[b].astype(int)), bgr(BLUE), 2, cv2.LINE_AA)
            if span[f]:
                trail = camera.project(pts[max(0, f - 90):f + 1])
                trail = trail[np.all(np.isfinite(trail), axis=1)].astype(np.int32)
                if len(trail) > 1:
                    cv2.polylines(img, [trail], False, bgr(YELLOW), 2, cv2.LINE_AA)
                p = camera.project(pts[f])
                if np.all(np.isfinite(p)):
                    cv2.circle(img, (int(p[0]), int(p[1])), 8, bgr(YELLOW), 3, cv2.LINE_AA)
            if c["gt_valid"][f]:
                g = camera.project(c["gt"][f])
                if np.all(np.isfinite(g)):
                    cv2.drawMarker(img, (int(g[0]), int(g[1])), bgr(AQUA), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
            est = track.estimates[f]
            lines = [f"Cam{cam}  frame {f}  REAL-TIME CAB CENTER"]
            if span[f]:
                q = pts[f] / upm
                label = {MEASURED: f"MEASURED ({est.cameras_used} cams)",
                         PARTIAL: f"PARTIAL (occluded, {est.occluded_edges} edges)",
                         PREDICTED: f"PREDICTED ({est.frames_since_measurement} frames)"}[status[f]]
                lines += [f"X {q[0]:.1f}  Y {q[1]:.1f}  Z {q[2]:.1f} m",
                          f"speed {speed[f]:.1f} m/s ({speed[f] * 3.6:.0f} km/h)", label]
                if np.isfinite(err[f]):
                    lines.append(f"error vs target {err[f]:.2f} m")
            else:
                lines.append("initializing")
            lines.append(video_note)
            cv2.rectangle(img, (10, 10), (560, 20 + 30 * len(lines)), (20, 20, 20), -1)
            for i, text in enumerate(lines):
                color = bgr(ORANGE) if text.startswith(("PREDICTED", "PARTIAL")) else (255, 255, 255)
                cv2.putText(img, text, (20, 38 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            legend = [("truck model (tracker)" if model is not None else "truck box (tracker)", BLUE), ("cab center + past trail", YELLOW), ("detector box", MUTED),
                      ("target CSV", AQUA)]
            for i, (text, col) in enumerate(legend):
                y = IMAGE_HEIGHT - 20 - 26 * (len(legend) - 1 - i)
                cv2.rectangle(img, (16, y - 12), (32, y + 2), bgr(col), -1)
                cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            ffmpeg.stdin.write(img.tobytes())
        cap.release()
        ffmpeg.stdin.close()
        ffmpeg.wait()
        log.info("видео: %s", target_file)


# ---------------------------------------------------------------- запуск

def run(args) -> None:
    setup_logging(args.out)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if args.video and "video" not in steps:
        steps.append("video")
    unknown = set(steps) - set(ALL_STEPS) - {"video"}
    if unknown:
        raise SystemExit(f"Неизвестные шаги: {', '.join(sorted(unknown))}. Доступны: {', '.join(ALL_STEPS)}, video")
    log.info("шаги: %s", ", ".join(steps))

    if "detect" in steps:
        run_detection(args)
    detections = load_detections(args.out)

    needs_center = any(s in steps for s in ("center", "optimize", "cab", "compare", "video"))
    center_result = step_center(args, detections) if needs_center else None
    center = center_result["spline"] if center_result else None

    optimized = None
    if "optimize" in steps:
        optimized = step_optimize(args, detections, center)
    elif (args.out / "optimized_spline.npz").exists():
        optimized = load_spline(args.out / "optimized_spline.npz")

    cab = None
    if "cab" in steps:
        cab = step_cab(args, detections, center)
    elif (args.out / "cab_spline.npz").exists() and any(s in steps for s in ("compare", "video")):
        data = np.load(args.out / "cab_spline.npz")
        observations = BoxObservations.from_detections([d for cam in detections.values() for d in cam.values()],
                                                       frame_offset=args.frame_offset)
        trajectory = load_spline(args.out / "cab_spline.npz")
        cab = {"trajectory": trajectory, "dimensions": TruckDimensions(*data["dimensions"]),
               "center_trajectory": offset_point_trajectory(trajectory, args.cab_center_back, args.cab_center_up),
               "edge_errors_px": np.full((len(observations.times), 4), np.nan), "converged": False, "nfev": 0,
               "observations": observations}

    realtime = None
    if any(s in steps for s in ("realtime", "compare", "video")):
        realtime = step_realtime(args, detections)

    if "compare" in steps or "video" in steps:
        if cab is None or optimized is None:
            raise SystemExit("Для compare/video нужны результаты шагов optimize и cab")
        metrics, ctx = evaluate_methods(args, detections, center_result, optimized, cab, realtime)
        ctx["metrics"] = metrics
        if "compare" in steps:
            (args.out / "compare").mkdir(parents=True, exist_ok=True)
            plot_comparison(args.out / "compare", ctx, cab, args)
            plot_camera_coverage(args.out / "compare", ctx["gt"], args, detections)
            write_report(args, metrics, cab, ctx)
            (args.out / "metrics.json").write_text(json.dumps(
                {"main_method": MAIN_METHOD, "dimensions_m": (cab["dimensions"].as_array() / args.units_per_meter).tolist(),
                 "methods": metrics}, ensure_ascii=False, indent=2))
            log.info("отчёт: %s", args.out / "report.md")
        if "video" in steps:
            render_videos(args, detections, ctx)
