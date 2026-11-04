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
  approx    базовый вариант: «обычная аппроксимация» 3D-точек триангуляции полиномом
            (причинный по окну — реальное время; глобальный — офлайн)       reconstruction/approximation.py
  compare   все варианты против CSV цели: <out>/report.md, metrics.json, metrics_summary.csv,
            compare/*.png, сводка в консоль
  video     видео с силуэтом модели и точкой кабины трекера: <out>/videos (по умолчанию включено,
            --no-video чтобы пропустить)

center, optimize и cab обрабатывают весь видеоряд сразу (используют будущие кадры) и нужны
только для сравнения. center при этом нужен всегда: его сплайн — начальное приближение для cab,
а его 3D-точки — вход для approx.
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
    AQUA, BLUE, CAMERA_COLORS, CAMERAS, GRID, INK, INK_2, MAGENTA, MUTED, NUM_VIDEO_FRAMES, ORANGE, YELLOW,
    _style, debiased_norm, find_gaps, run_evaluation, set_cameras, stats, truck_frame_components,
)
from pipeline.coverage import plot_camera_coverage
from pipeline.detect import detections_dir, detector_info, load_detections, run_detection
from pipeline.export import tracker_rows, write_trajectory_csv
from reconstruction.approximation import causal_polynomial, global_polynomial
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
from reconstruction.data_io import VideoResolutionError, load_target, make_camera, resolve_image_sizes, video_path
from reconstruction.objective import ObjectiveData
from reconstruction.optimizer import optimize_trajectory
from reconstruction.truck_model import SCENE_SCALE, TruckModel
from reconstruction.visibility import INITIALIZING, MEASURED, PARTIAL, PREDICTED, TrackerSettings, run_tracker

log = logging.getLogger("peleng")
ALL_STEPS = ["detect", "center", "optimize", "cab", "realtime", "approx", "compare"]
MAIN_METHOD = "realtime"
PURPLE, BROWN = "#7b5cd6", "#a0522d"
METHOD_NAMES = {
    "triangulation": "Офлайн. Центр маски: триангуляция",
    "center": "Офлайн. Центр маски: начальный сплайн",
    "approx_poly": "Офлайн. Центр маски: полином по всем 3D-точкам",
    "optimized": "Офлайн. Центр маски: оптимизация по пикселям",
    "cab": "Офлайн. Кабина: низ передней грани",
    "cab_center": "Офлайн. Кабина: центр кабины",
    "approx_rt": "Реальное время: полином по окну на 3D-точках",
    "realtime_ref": "Реальное время: низ передней грани",      # с 3D-моделью — «точка CSV цели на модели»
    "realtime": "Реальное время: центр кабины (основной)",
}
METHOD_COLORS = {"triangulation": ORANGE, "center": MUTED, "approx_poly": BROWN, "optimized": MAGENTA, "cab": INK_2,
                 "cab_center": AQUA, "approx_rt": PURPLE, "realtime_ref": YELLOW, "realtime": BLUE}
REALTIME_METHODS = ("approx_rt", "realtime_ref", "realtime")


def build_main_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"), help="веса YOLO (.pt)")
    p.add_argument("--data", type=Path, default=Path("tmp/project2"), help="папка с видео и CSV")
    p.add_argument("--out", type=Path, default=Path("outputs/main"))
    p.add_argument("--steps", default=",".join(ALL_STEPS), help=f"через запятую из: {', '.join(ALL_STEPS)}")
    p.add_argument("--video", action="store_true", help="добавить шаг video (при --steps)")
    p.add_argument("--no-video", action="store_true", help="не делать видео при полном прогоне")
    p.add_argument("--cameras", default=None,
                   help="номера камер через запятую (по умолчанию 1,2,3,4,5): файлы ...CineCameraActor{N}")
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
    g.add_argument("--backend", default="auto", choices=["auto", "torch", "onnx", "openvino"],
                   help="бэкенд инференса (auto: GPU — torch, CPU — из профиля производительности)")
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

    g = p.add_argument_group("approx (обычная аппроксимация 3D-точек)")
    g.add_argument("--approx-window", type=int, default=15, help="окно причинного полинома, кадров")
    g.add_argument("--approx-degree", type=int, default=2, help="степень причинного полинома")
    g.add_argument("--approx-min-points", type=int, default=8, help="минимум точек в окне")
    g.add_argument("--approx-max-extrapolation", type=int, default=30,
                   help="сколько кадров держать прогноз без точек")
    g.add_argument("--approx-huber", type=float, default=100.0, help="порог Хьюбера, ед. мира")
    g.add_argument("--approx-poly-degree", type=int, default=8, help="степень офлайн-полинома по всем точкам")
    return p


def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", encoding="utf-8")],
    )


def camera_provider(args):
    """Камера (номер, кадр CSV) с разрешением её видео (args.image_sizes, см. run)."""
    return lambda cam, frame: make_camera(args.data, cam, frame, args.image_sizes[cam])


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
        args.image_sizes, frame_offset=args.frame_offset,
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
    observations = BoxObservations.from_detections(rows, args.image_sizes, frame_offset=args.frame_offset)
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
    make = camera_provider(args)

    def provider(cam, frame):
        if (cam, frame) not in cache:
            cache[(cam, frame)] = make(cam, frame)
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
    cams_by_frame: dict = {}
    for d in rows:
        cams_by_frame.setdefault(d.frame, []).append(d.camera)
    n = write_trajectory_csv(args.out / "trajectory.csv", tracker_rows(result, np.arange(NUM_VIDEO_FRAMES), args.fps,
                                                                       args.units_per_meter, cams_by_frame))
    log.info("координаты трекера: %s (%d кадров, центр кабины, м)", args.out / "trajectory.csv", n)
    return {"result": result, "dimensions": dims, "model": model, "ms_per_frame": elapsed / NUM_VIDEO_FRAMES * 1000}


def step_approx(args, center_result) -> dict:
    """
    Базовый вариант без модели объекта: полином по 3D-точкам триангуляции центров масок.
    Причинный (по окну прошлых кадров) — реальное время; глобальный — офлайн.
    """
    ctx = center_result["ctx"]
    frames = ctx["frames"].astype(float)
    points = np.where(ctx["ok"][:, None], ctx["tri_points"], np.nan)
    started = time.time()
    rt = causal_polynomial(frames, points, window=args.approx_window, degree=args.approx_degree,
                           min_points=args.approx_min_points, max_extrapolation=args.approx_max_extrapolation,
                           huber_scale=args.approx_huber)
    ms_per_frame = (time.time() - started) / len(frames) * 1000
    poly = global_polynomial(frames, points, degree=args.approx_poly_degree)
    log.info("аппроксимация: окно %d кадров, степень %d, %.2f мс/кадр; статусы %s; офлайн-полином степени %d",
             args.approx_window, args.approx_degree, ms_per_frame,
             {k: int((rt.status == k).sum()) for k in ("measured", "predicted", "none")}, args.approx_poly_degree)
    np.savez(args.out / "approx_track.npz", frames=frames, rt_points=rt.points, rt_velocity=rt.velocity,
             rt_status=rt.status, rt_points_used=rt.points_used, poly_points=poly.points, poly_velocity=poly.velocity,
             window=args.approx_window, degree=args.approx_degree, poly_degree=args.approx_poly_degree)
    return {"rt": rt, "poly": poly, "ms_per_frame": ms_per_frame}


# ---------------------------------------------------------------- сравнение

def trajectory_acceleration(points: np.ndarray, fps: float, upm: float) -> np.ndarray:
    """
    Ускорение по самой траектории (вторая разность положений), м/с². В отличие от скачка скорости
    фильтра видит и «телепортацию» — сдвиг положения поправкой измерения в обход скорости.
    """
    acc = np.full(len(points), np.nan)
    acc[1:-1] = np.linalg.norm(points[2:] - 2 * points[1:-1] + points[:-2], axis=1) * fps ** 2 / upm
    return acc


def _axis_metrics(diff_m: np.ndarray, mask: np.ndarray) -> dict:
    """Ошибка по осям мира X / Y / Z: смещение (среднее), MAE, RMSE, м."""
    out = {}
    for i, axis in enumerate("xyz"):
        d = diff_m[mask, i]
        out[axis] = {"bias": float(np.mean(d)), "mae": float(np.mean(np.abs(d))),
                     "rmse": float(np.sqrt(np.mean(d ** 2)))} if len(d) else {}
    return out


def evaluate_methods(args, detections, center_result, optimized, cab, realtime, approx) -> tuple[dict, dict]:
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
    poly, rt = approx["poly"], approx["rt"]
    series["approx_poly"] = {"points": poly.points, "spline": None,
                             "speed": np.linalg.norm(poly.velocity, axis=1) * fps / upm}
    series["approx_rt"] = {"points": rt.points, "spline": None, "speed": np.linalg.norm(rt.velocity, axis=1) * fps / upm,
                           "status": rt.status}
    track = realtime["result"]
    rt_speed = track.array("speed_mps")
    rt_status = track.array("status")
    series["realtime_ref"] = {"points": track.array("position"), "spline": None, "speed": rt_speed, "status": rt_status}
    series["realtime"] = {"points": track.array("cab_center"), "spline": None, "speed": rt_speed, "status": rt_status}
    # порядок — как в METHOD_NAMES (отчёт, сводка и графики идут в этом порядке)
    series = {name: series[name] for name in METHOD_NAMES if name in series}

    metrics = {}
    for name, s in series.items():
        pts = s["points"]
        has = np.all(np.isfinite(pts), axis=1)
        diff_m = (pts - gt) / upm
        err = np.linalg.norm(diff_m, axis=1)
        along, cross, vertical = truck_frame_components(diff_m, gt_vel)
        comp_mask = has & gt_valid & moving
        debiased, bias = debiased_norm(along, cross, vertical, comp_mask)
        evaluated = has & gt_valid
        m = {
            "realtime": name in REALTIME_METHODS,
            "frames": int(evaluated.sum()),
            "availability": float(has.sum() / NUM_VIDEO_FRAMES),
            "position_m": stats(err[evaluated]),
            "position_mae_m": float(np.mean(err[evaluated])) if evaluated.any() else float("nan"),
            "per_axis_m": _axis_metrics(diff_m, evaluated),
            "bias_m": bias,
            "position_without_bias_m": stats(debiased[comp_mask]),
            "by_detected_cameras": {int(k): stats(err[evaluated & (detected_cams == k)])
                                    for k in range(1, len(CAMERAS) + 1) if np.any(evaluated & (detected_cams == k))},
        }
        if "status" in s:
            m["by_status"] = {str(k): stats(err[evaluated & (s["status"] == k)])
                              for k in np.unique(s["status"]) if np.any(evaluated & (s["status"] == k))}
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
            traj_acc = trajectory_acceleration(pts, fps, upm)
            path = float(np.nansum(np.linalg.norm(np.diff(pts[span], axis=0), axis=1)) / upm)
            gt_path = float(np.sum(np.linalg.norm(np.diff(gt_spline.evaluate(csv_t[span]), axis=0), axis=1)) / upm)
            m["kinematics"] = {
                "speed_max_mps": float(np.nanmax(speed)) if np.any(np.isfinite(speed)) else float("nan"),
                "speed_error_mps": stats(np.abs(speed - gt_speed)[span]),
                "speed_change_max_mps2": float(jumps.max()) if len(jumps) else float("nan"),
                "speed_change_p99_mps2": float(np.percentile(jumps, 99)) if len(jumps) else float("nan"),
                "trajectory_accel_mps2": stats(traj_acc[np.isfinite(traj_acc)]),
                "path_m": path, "target_path_m": gt_path,
            }
        s["error"] = err
        metrics[name] = m

    # Погрешность трекера: заявленное СКО положения (по горизонтали) против фактической ошибки
    std_m = track.array("position_std") / upm
    horiz = np.linalg.norm((track.array("position") - gt)[:, :2], axis=1) / upm
    ok_std = np.isfinite(std_m) & np.isfinite(horiz) & gt_valid
    for name in ("realtime_ref", "realtime"):
        metrics[name]["uncertainty"] = {
            "position_std_median_m": float(np.median(std_m[ok_std])) if ok_std.any() else float("nan"),
            "horizontal_error_median_m": float(np.median(horiz[ok_std])) if ok_std.any() else float("nan"),
            "error_to_std_ratio_median": float(np.median(horiz[ok_std] / np.maximum(std_m[ok_std], 1e-6)))
            if ok_std.any() else float("nan"),
            "within_2std_fraction": float(np.mean(horiz[ok_std] <= 2 * std_m[ok_std])) if ok_std.any() else float("nan"),
            "note": "СКО — по центру модели из ковариации фильтра, ошибка — точки вывода (с ошибкой курса); "
                    "идеальная калибровка: отношение ~1, в 2σ ~95 %",
        }
        metrics[name]["ms_per_frame"] = realtime["ms_per_frame"]
    metrics["approx_rt"]["ms_per_frame"] = approx["ms_per_frame"]

    target_jumps = np.abs(np.diff(gt_speed)) * fps
    target_acc = trajectory_acceleration(gt, fps, upm)
    ctx_out = dict(frames=frames, gt=gt, gt_valid=gt_valid, gt_speed=gt_speed, series=series,
                   detected_cams=detected_cams, target_issues=target.issues,
                   target_speed_change_p99=float(np.percentile(target_jumps, 99)), realtime=realtime, approx=approx,
                   std_m=std_m, horiz_err_m=horiz,
                   target_trajectory_accel=stats(target_acc[np.isfinite(target_acc)]))
    return metrics, ctx_out


def plot_comparison(out: Path, c: dict, cab: dict, args) -> None:
    plt = _style()
    upm, frames = args.units_per_meter, c["frames"]
    note = detector_info(args.out)["note"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in ("center", "optimized", "cab_center", "approx_rt", "realtime"):
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
    a1.set_title("Скорость: офлайн — производная сплайна, реальное время — вектор скорости фильтра")
    a1.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "03_speed.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(17, 5))
    names = [n for n in METHOD_NAMES if n in c["metrics"]]
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

    # 07. Координаты от времени (ТЗ п.5): X, Y, Z цели и оценок; справа — разность с целью
    seconds = frames / args.fps
    shown = ("realtime", "approx_rt", "cab_center")
    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True, gridspec_kw={"width_ratios": [3, 2]})
    for i, axis in enumerate("XYZ"):
        a_val, a_err = axes[i]
        a_val.plot(seconds, gt[:, i], color=INK_2, lw=2.2, label="цель (CSV)")
        for name in shown:
            p = c["series"][name]["points"] / upm
            a_val.plot(seconds, p[:, i], color=METHOD_COLORS[name], lw=2.2 if name == MAIN_METHOD else 1.3,
                       ls="-" if name in REALTIME_METHODS else "--", label=METHOD_NAMES[name])
            a_err.plot(seconds, p[:, i] - gt[:, i], color=METHOD_COLORS[name],
                       lw=1.8 if name == MAIN_METHOD else 1.1, ls="-" if name in REALTIME_METHODS else "--")
        a_err.axhline(0, color=MUTED, lw=1)
        a_val.set_ylabel(f"{axis}, м")
        a_err.set_ylabel(f"Δ{axis}, м")
        lim = np.nanpercentile(np.abs(np.concatenate(
            [c["series"][n]["points"][:, i] / upm - gt[:, i] for n in shown])), 98)
        if np.isfinite(lim) and lim > 0:
            a_err.set_ylim(-1.5 * lim, 1.5 * lim)
    axes[0, 0].set_title("Координаты от времени: цель и оценки (сплошные — реальное время, пунктир — офлайн)")
    axes[0, 1].set_title("Разность оценка − цель (обрезано по p98)")
    axes[0, 0].legend(loc="upper left", fontsize=8, ncol=2)
    for a in axes[-1]:
        a.set_xlabel("Время, с")
    sec = axes[0, 0].secondary_xaxis("top", functions=(lambda s_: s_ * args.fps, lambda f_: f_ / args.fps))
    sec.set_xlabel("Кадр видео", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "07_coordinates_vs_time.png")
    plt.close(fig)

    # 08. «Обычная аппроксимация» 3D-точек против трекера
    rt = c["approx"]["rt"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 1.3]})
    a1.scatter(frames, c["series"]["triangulation"]["error"], s=5, color=ORANGE, alpha=0.5, lw=0,
               label=METHOD_NAMES["triangulation"])
    for name in ("approx_poly", "center", "approx_rt", "realtime_ref"):
        a1.plot(frames, c["series"][name]["error"], color=METHOD_COLORS[name], lw=2 if name == "approx_rt" else 1.4,
                ls="-" if name in REALTIME_METHODS else "--", label=METHOD_NAMES[name])
    predicted = rt.status == "predicted"
    for a, b in find_gaps(~predicted):
        a1.axvspan(a - 0.5, b + 0.5, color=PURPLE, alpha=0.12, lw=0)
    for a, b in find_gaps(rt.status != "none"):
        a1.axvspan(a - 0.5, b + 0.5, color=GRID, alpha=0.9, lw=0)
    a1.set_ylim(0, None)
    a1.set_ylabel("Ошибка относительно CSV цели, м")
    a1.set_xlabel("Кадр видео (сиреневым — прогноз полинома без точек, серым — оценки нет)")
    a1.set_title(f"Аппроксимация 3D-точек полиномом (окно {args.approx_window} кадров, "
                 f"степень {args.approx_degree}) против модели грузовика — {note}")
    a1.legend(loc="upper left", fontsize=8)
    gt_m = gt / upm
    a2.plot(gt_m[:, 0], gt_m[:, 1], color=INK_2, lw=2.2, label="цель (CSV)")
    tp = c["series"]["triangulation"]["points"] / upm
    a2.scatter(tp[:, 0], tp[:, 1], s=6, color=ORANGE, alpha=0.5, lw=0, label="3D-точки (центр маски)")
    for name in ("approx_poly", "approx_rt", "realtime_ref"):
        p = c["series"][name]["points"] / upm
        a2.plot(p[:, 0], p[:, 1], color=METHOD_COLORS[name], lw=1.6, ls="-" if name in REALTIME_METHODS else "--",
                label=METHOD_NAMES[name])
    a2.set_aspect("equal")
    a2.set_xlabel("X, м")
    a2.set_ylabel("Y, м")
    a2.set_title("Вид сверху: 3D-точки и их аппроксимации")
    a2.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "08_approximation_vs_tracker.png")
    plt.close(fig)

    # 09. Погрешность трекера: заявленное СКО против фактической ошибки (ТЗ п.6)
    track = c["realtime"]["result"]
    status = track.array("status")
    std, herr = c["std_m"], c["horiz_err_m"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [3, 1.2]})
    a1.fill_between(frames, 0, 2 * std, color=BLUE, alpha=0.15, lw=0, label="2·СКО фильтра (по горизонтали)")
    a1.plot(frames, herr, color=YELLOW, lw=1.5, label=f"{METHOD_NAMES['realtime_ref']}: ошибка по горизонтали")
    for code, color in ((PREDICTED, ORANGE), (PARTIAL, MAGENTA)):
        for a, b in find_gaps(status != code):
            a1.axvspan(a - 0.5, b + 0.5, color=color, alpha=0.18, lw=0)
    a1.set_ylim(0, None)
    a1.set_xlabel("Кадр видео (оранжевым — predicted, розовым — partial)")
    a1.set_ylabel("м")
    a1.set_title("Оценка погрешности трекера: фактическая ошибка и заявленная неопределённость")
    a1.legend(loc="upper left", fontsize=8)
    ok_std = np.isfinite(std) & np.isfinite(herr)
    a2.scatter(std[ok_std], herr[ok_std], s=8, color=BLUE, alpha=0.5, lw=0)
    top = max(np.nanpercentile(std[ok_std], 99), np.nanpercentile(herr[ok_std], 99)) if ok_std.any() else 1
    a2.plot([0, top], [0, 2 * top], color=MUTED, lw=1, ls="--", label="ошибка = 2·СКО")
    a2.set_xlim(0, top)
    a2.set_ylim(0, top * 2)
    a2.set_xlabel("СКО фильтра, м")
    a2.set_ylabel("ошибка, м")
    a2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "09_realtime_uncertainty.png")
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
        "| ускорение по траектории, медиана / p95 / макс, м/с² | путь, м (цель) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        b = m["bias_m"]
        k = m.get("kinematics")
        kin = (f"{k['speed_max_mps']:.1f} | {k['speed_error_mps']['median']:.2f} / {k['speed_error_mps']['p95']:.2f} "
               f"| {k['speed_change_p99_mps2']:.1f} / {k['speed_change_max_mps2']:.1f} "
               f"| {k['trajectory_accel_mps2']['median']:.1f} / {k['trajectory_accel_mps2']['p95']:.1f} / "
               f"{k['trajectory_accel_mps2']['max']:.0f} "
               f"| {k['path_m']:.0f} ({k['target_path_m']:.0f})") if k else "— | — | — | — | —"
        title = f"**{METHOD_NAMES[name]}**" if name == MAIN_METHOD else METHOD_NAMES[name]
        lines.append(f"| {title} | {m['frames']} | {_fmt(m['position_m'])} | {_fmt(m['position_without_bias_m'])} "
                     f"| {b['along_m']:+.2f} / {b['cross_m']:+.2f} / {b['vertical_m']:+.2f} | {kin} |")
    rt = c["realtime"]["result"]
    status = rt.array("status")
    since = rt.array("frames_since_measurement")
    lines += ["", f"Скорость цели по CSV: максимум {np.nanmax(c['gt_speed']):.1f} м/с; скачок скорости между кадрами p99 "
              f"{c['target_speed_change_p99']:.1f} м/с²; ускорение по траектории CSV (вторая разность, с шумом "
              f"округления CSV) медиана / p95 {c['target_trajectory_accel']['median']:.1f} / "
              f"{c['target_trajectory_accel']['p95']:.1f} м/с². «Скачок скорости» у трекера считается по его "
              "скорости и не видит сдвигов положения поправкой; «ускорение по траектории» — видит.", "",
              "## Ошибка по осям мира, м (смещение / MAE / RMSE)", "",
              "| Вариант | доля кадров с оценкой | X | Y | Z | MAE 3D |", "|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        ax = m["per_axis_m"]
        cells = " | ".join(f"{ax[a]['bias']:+.2f} / {ax[a]['mae']:.2f} / {ax[a]['rmse']:.2f}" if ax.get(a) else "—"
                           for a in "xyz")
        title = f"**{METHOD_NAMES[name]}**" if name == MAIN_METHOD else METHOD_NAMES[name]
        lines.append(f"| {title} | {m['availability'] * 100:.0f} % | {cells} | {m['position_mae_m']:.2f} |")
    u = metrics["realtime_ref"]["uncertainty"]
    lines += ["", "## Оценка погрешности реального времени", "",
              f"Фильтр заявляет СКО положения по горизонтали: медиана {u['position_std_median_m']:.2f} м; фактическая "
              f"ошибка по горизонтали ({METHOD_NAMES['realtime_ref']}): медиана {u['horizontal_error_median_m']:.2f} м. "
              f"Отношение ошибка / СКО: медиана {u['error_to_std_ratio_median']:.1f}; в пределах 2·СКО "
              f"{u['within_2std_fraction'] * 100:.0f} % кадров (при честной ковариации ~95 %). "
              "Отношение больше ~2 означает, что ковариация занижена: ошибка курса и несовпадение модели с рамками "
              "в ней не учтены (см. compare/09_realtime_uncertainty.png).", "",
              "Ошибка основного варианта по статусу кадра, м (медиана / p95 / RMSE / максимум):", "",
              "| статус | кадров | ошибка |", "|---|---|---|"]
    for code, s in metrics[MAIN_METHOD].get("by_status", {}).items():
        lines.append(f"| {code} | {s['n']} | {_fmt(s)} |")
    a = c["approx"]
    lines += ["", "## Базовый вариант: обычная аппроксимация 3D-точек", "",
              f"Полином степени {args.approx_degree} по окну {args.approx_window} кадров (только прошлое, Хьюбер "
              f"{args.approx_huber / upm:.1f} м) на точках триангуляции центров масок; без точек — прогноз последним "
              f"полиномом по прямой до {args.approx_max_extrapolation} кадров; {a['ms_per_frame']:.2f} мс/кадр. "
              f"Офлайн — один полином степени {args.approx_poly_degree} по всем точкам. "
              "Оба наследуют смещение центра маски относительно точки CSV (столбец «без постоянного смещения» "
              "показывает ошибку формы) и скачки при смене камер; у них нет формы грузовика и курса, "
              "поэтому точку кабины из них получить нельзя. См. compare/08_approximation_vs_tracker.png.", "",
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
    lines += ["", "Отчёт базового варианта (центр маски): [center/report.md](center/report.md); "
              "сводная таблица: [metrics_summary.csv](metrics_summary.csv)", "",
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


SUMMARY_COLUMNS = ["method", "realtime", "frames", "availability", "median_m", "p95_m", "rmse_m", "max_m", "mae_m",
                   "debiased_median_m", "rmse_x_m", "rmse_y_m", "rmse_z_m", "speed_error_median_mps",
                   "speed_change_max_mps2", "ms_per_frame"]


def summary_rows(metrics: dict) -> list[dict]:
    """Одна строка на вариант — для CSV и консоли."""
    rows = []
    for name, m in metrics.items():
        p, k, ax = m["position_m"], m.get("kinematics"), m["per_axis_m"]
        rows.append({
            "method": METHOD_NAMES[name], "realtime": "да" if m["realtime"] else "офлайн", "frames": m["frames"],
            "availability": round(m["availability"], 3),
            "median_m": p.get("median"), "p95_m": p.get("p95"), "rmse_m": p.get("rmse"), "max_m": p.get("max"),
            "mae_m": m["position_mae_m"], "debiased_median_m": m["position_without_bias_m"].get("median"),
            "rmse_x_m": ax["x"].get("rmse"), "rmse_y_m": ax["y"].get("rmse"), "rmse_z_m": ax["z"].get("rmse"),
            "speed_error_median_mps": k["speed_error_mps"].get("median") if k else None,
            "speed_change_max_mps2": k["speed_change_max_mps2"] if k else None,
            "ms_per_frame": m.get("ms_per_frame"),
        })
    return rows


def write_summary_csv(path: Path, metrics: dict) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for row in summary_rows(metrics):
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()})


def print_summary(metrics: dict, args) -> None:
    """Сводка метрик качества в консоль (ТЗ п.6): ошибка относительно CSV цели по вариантам."""
    def num(v, digits=2):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}"

    head = f"{'Вариант':62} {'РВ':6} {'кадров':>6} {'медиана':>8} {'p95':>7} {'RMSE':>7} {'макс':>7} " \
           f"{'без смещ.':>9} {'σv, м/с':>8} {'Δv макс':>8} {'мс/кадр':>8}"
    print()
    print("СВОДКА КАЧЕСТВА: ошибка положения относительно CSV цели, м (YOLO обучена на этих же видео)")
    print(head)
    print("-" * len(head))
    for r in summary_rows(metrics):
        mark = "*" if r["method"] == METHOD_NAMES[MAIN_METHOD] else " "
        print(f"{mark}{r['method']:61} {r['realtime']:6} {r['frames']:6d} {num(r['median_m']):>8} {num(r['p95_m']):>7} "
              f"{num(r['rmse_m']):>7} {num(r['max_m']):>7} {num(r['debiased_median_m']):>9} "
              f"{num(r['speed_error_median_mps']):>8} {num(r['speed_change_max_mps2'], 1):>8} {num(r['ms_per_frame']):>8}")
    u = metrics["realtime_ref"].get("uncertainty", {})
    print(f"* основной результат. Погрешность трекера: СКО {num(u.get('position_std_median_m'))} м, "
          f"ошибка/СКО {num(u.get('error_to_std_ratio_median'), 1)}, в 2σ {num(u.get('within_2std_fraction', 0) * 100, 0)} %")
    print(f"Отчёт: {args.out / 'report.md'}; графики: {args.out / 'compare'}; таблица: {args.out / 'metrics_summary.csv'}")
    print()


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
        if not video_path(args.data, cam).exists():
            log.warning("видео: нет файла %s, камера пропущена", video_path(args.data, cam))
            continue
        cap = cv2.VideoCapture(str(video_path(args.data, cam)))
        target_file = out_dir / f"realtime_cam{cam}.mp4"
        size = args.image_sizes[cam]                     # разрешение этого видео (см. run)
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
             "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-pix_fmt", "yuv420p", str(target_file)], stdin=subprocess.PIPE)
        for f in range(NUM_VIDEO_FRAMES):
            ok, img = cap.read()
            if not ok:
                break
            if (img.shape[1], img.shape[0]) != size:
                # кадр не того размера: приводим к размеру видео — K камеры построена для него
                log.warning("видео cam%d, кадр %d: %d×%d вместо %d×%d — кадр масштабирован", cam, f,
                            img.shape[1], img.shape[0], *size)
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            camera = make_camera(args.data, cam, f + args.frame_offset, size)
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
                y = img.shape[0] - 20 - 26 * (len(legend) - 1 - i)
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
    if args.cameras:
        set_cameras(int(c) for c in args.cameras.split(",") if c.strip())
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    full_run = set(ALL_STEPS) - {"detect"} <= set(steps)       # полный прогон (detect мог быть пропущен)
    if (args.video or (full_run and not args.no_video)) and "video" not in steps:
        steps.append("video")
    if args.no_video and "video" in steps:
        steps.remove("video")
    unknown = set(steps) - set(ALL_STEPS) - {"video"}
    if unknown:
        raise SystemExit(f"Неизвестные шаги: {', '.join(sorted(unknown))}. Доступны: {', '.join(ALL_STEPS)}, video")
    log.info("шаги: %s; камеры: %s", ", ".join(steps), CAMERAS)

    if "detect" in steps:
        run_detection(args)
    detections = load_detections(args.out)
    # Разрешение кадра — отдельно для каждой камеры: из её видео, без видео — из meta.json детекций
    try:
        args.image_sizes = resolve_image_sizes(args.data, CAMERAS, detections_dir(args.out))
    except VideoResolutionError as e:
        raise SystemExit(f"Разрешение видео: {e}") from None
    log.info("разрешение кадра по камерам: %s",
             ", ".join(f"cam{cam} {w}×{h}" for cam, (w, h) in sorted(args.image_sizes.items())))

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
                                                       args.image_sizes, frame_offset=args.frame_offset)
        trajectory = load_spline(args.out / "cab_spline.npz")
        cab = {"trajectory": trajectory, "dimensions": TruckDimensions(*data["dimensions"]),
               "center_trajectory": offset_point_trajectory(trajectory, args.cab_center_back, args.cab_center_up),
               "edge_errors_px": np.full((len(observations.times), 4), np.nan), "converged": False, "nfev": 0,
               "observations": observations}

    realtime = None
    if any(s in steps for s in ("realtime", "compare", "video")):
        realtime = step_realtime(args, detections)

    approx = None
    if any(s in steps for s in ("approx", "compare", "video")):
        approx = step_approx(args, center_result)

    if "compare" in steps or "video" in steps:
        if cab is None or optimized is None:
            raise SystemExit("Для compare/video нужны результаты шагов optimize и cab")
        metrics, ctx = evaluate_methods(args, detections, center_result, optimized, cab, realtime, approx)
        ctx["metrics"] = metrics
        if "compare" in steps:
            (args.out / "compare").mkdir(parents=True, exist_ok=True)
            plot_comparison(args.out / "compare", ctx, cab, args)
            plot_camera_coverage(args.out / "compare", ctx["gt"], args, detections)
            write_report(args, metrics, cab, ctx)
            (args.out / "metrics.json").write_text(json.dumps(
                {"main_method": MAIN_METHOD, "dimensions_m": (cab["dimensions"].as_array() / args.units_per_meter).tolist(),
                 "detector_trained_on_these_videos": True, "methods": metrics}, ensure_ascii=False, indent=2))
            write_summary_csv(args.out / "metrics_summary.csv", metrics)
            log.info("отчёт: %s", args.out / "report.md")
            print_summary(metrics, args)
        if "video" in steps:
            render_videos(args, detections, ctx)
