"""
Шаг center (офлайн, базовый вариант): точка объекта — центр маски YOLO.

Триангуляция по кадрам -> начальный B-сплайн -> метрики против CSV цели -> графики и
<out>/center/report.md. Этот сплайн — начальное приближение для офлайн-модели кабины.
Здесь же проверка смещения кадров видео ↔ CSV (OFFSETS_TO_SCAN) и общие для отчётов
функции: статистики, разложение ошибки в осях грузовика, палитра графиков.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detection.schema import Detection
from reconstruction.bspline import fit_initial_bspline
from reconstruction.data_io import IMAGE_HEIGHT, IMAGE_WIDTH, load_target, make_camera
from reconstruction.triangulation import triangulate_robust

CAMERAS = [1, 2, 3, 4, 5]        # список меняется на месте (set_cameras), чтобы все модули видели один объект
NUM_VIDEO_FRAMES = 1001


def set_cameras(ids) -> None:
    """Задать номера камер (файлы object_transform_CineCameraActor{N}.csv и видео {N}); список общий для всех шагов."""
    ids = sorted({int(i) for i in ids})
    if not ids:
        raise ValueError("Нужна хотя бы одна камера")
    CAMERAS[:] = ids
OFFSETS_TO_SCAN = range(-3, 4)
log = logging.getLogger("peleng")

# Палитра графиков (светлая тема); порядок цветов камер фиксирован, чтобы камера везде была одного цвета
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CAMERA_COLORS = {1: BLUE, 2: ORANGE, 3: AQUA, 4: YELLOW, 5: MAGENTA}


# ---------------------------------------------------------------- геометрия

class CameraCache:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._cache = {}

    def get(self, cam: int, csv_frame: int):
        key = (cam, csv_frame)
        if key not in self._cache:
            try:
                self._cache[key] = make_camera(self.data_dir, cam, csv_frame)
            except ValueError:
                self._cache[key] = None
        return self._cache[key]


@dataclass
class FrameTriangulation:
    frame: int
    cams_detected: list[int]
    cams_used: list[int]                     # надёжные детекции, переданные в триангуляцию
    inliers: list[int] = field(default_factory=list)
    point: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    reprojection_px: dict[int, float] = field(default_factory=dict)
    ok: bool = False


def triangulate_frames(detections, cameras: CameraCache, offset: int, max_reproj_px: float) -> list[FrameTriangulation]:
    out = []
    for frame in range(NUM_VIDEO_FRAMES):
        dets = {cam: detections[cam][frame] for cam in CAMERAS if frame in detections[cam]}
        used = [cam for cam, d in dets.items() if not d.unreliable and cameras.get(cam, frame + offset) is not None]
        rec = FrameTriangulation(frame, sorted(dets), used)
        if len(used) >= 2:
            res = triangulate_robust(
                [cameras.get(c, frame + offset) for c in used],
                [(dets[c].point_u, dets[c].point_v) for c in used],
                max_reprojection_error_px=max_reproj_px,
            )
            rec.ok = res.ok
            rec.point = res.point
            rec.inliers = [c for c, keep in zip(used, res.inliers) if keep]
            rec.reprojection_px = {c: float(e) for c, e in zip(used, res.reprojection_errors)}
        out.append(rec)
    return out


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "max": float(values.max()),
    }


def find_gaps(mask: np.ndarray) -> list[tuple[int, int]]:
    """Отрезки [start, end] подряд идущих кадров, где mask = False."""
    gaps, start = [], None
    for i, m in enumerate(mask):
        if not m and start is None:
            start = i
        if m and start is not None:
            gaps.append((start, i - 1))
            start = None
    if start is not None:
        gaps.append((start, len(mask) - 1))
    return gaps


def truck_frame_components(diff_m: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Разложение разности (N, 3) на вдоль движения / поперёк (горизонтально) / вертикаль."""
    heading = velocity[:, :2] / np.maximum(np.linalg.norm(velocity[:, :2], axis=1, keepdims=True), 1e-9)
    along = np.sum(diff_m[:, :2] * heading, axis=1)
    cross = diff_m[:, 0] * -heading[:, 1] + diff_m[:, 1] * heading[:, 0]
    return along, cross, diff_m[:, 2]


def debiased_norm(along, cross, vertical, mask) -> tuple[np.ndarray, dict]:
    """Ошибка после вычитания среднего смещения в осях грузовика (оценённого по mask)."""
    bias = {"along_m": float(np.mean(along[mask])), "cross_m": float(np.mean(cross[mask])),
            "vertical_m": float(np.mean(vertical[mask]))}
    norm = np.sqrt((along - bias["along_m"]) ** 2 + (cross - bias["cross_m"]) ** 2
                   + (vertical - bias["vertical_m"]) ** 2)
    return norm, bias


# ---------------------------------------------------------------- оценка

def run_evaluation(args, detections: dict[int, dict[int, Detection]]) -> dict:
    out: Path = args.out
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    upm = args.units_per_meter
    target = load_target(args.data)
    for issue in target.issues:
        log.warning("CSV цели: испорченная ячейка: %s — кадр исключён из оценки", issue)
    cameras = CameraCache(args.data)

    # Сплайн по самой цели (в кадрах CSV) — только для направления движения при оценке
    gt_spline = fit_initial_bspline(target.frames[target.valid].astype(float), target.positions[target.valid],
                                    knot_spacing=5)
    min_speed = args.units_per_meter / args.fps          # 1 м/с в единицах мира за кадр

    # --- проверка смещения кадров видео ↔ CSV.
    # Центр маски и точка цели — разные точки кузова, поэтому сырая ошибка содержит
    # постоянное смещение вдоль движения, и сдвиг по времени его частично «компенсирует».
    # Сравниваем смещения по ошибке после вычитания постоянного смещения в осях грузовика.
    offset_scan = {}
    for k in OFFSETS_TO_SCAN:
        tri_k = [t for t in triangulate_frames(detections, cameras, k, args.max_reproj_px)
                 if t.ok and 0 <= t.frame + k <= target.frames[-1]]
        csv_frames = np.array([t.frame + k for t in tri_k], dtype=float)
        gt_k = np.array([target.position(int(f)) for f in csv_frames])
        vel_k = gt_spline.derivative().evaluate(np.clip(csv_frames, gt_spline.t_start, gt_spline.t_end))
        diff = (np.array([t.point for t in tri_k]) - gt_k) / upm
        usable = np.all(np.isfinite(diff), axis=1) & (np.linalg.norm(vel_k, axis=1) > min_speed)
        comps = truck_frame_components(diff, vel_k)
        norm, _ = debiased_norm(*comps, usable)
        offset_scan[k] = {
            "median_m": float(np.median(np.linalg.norm(diff[usable], axis=1))),
            "median_debiased_m": float(np.median(norm[usable])),
            "frames": int(usable.sum()),
        }
        log.info("смещение %+d: медиана ошибки %.3f м, без постоянного смещения %.3f м (кадров %d)",
                 k, offset_scan[k]["median_m"], offset_scan[k]["median_debiased_m"], offset_scan[k]["frames"])
    offset = args.frame_offset
    best_offset = min(offset_scan, key=lambda o: offset_scan[o]["median_debiased_m"])

    tri = triangulate_frames(detections, cameras, offset, args.max_reproj_px)
    frames = np.arange(NUM_VIDEO_FRAMES)
    gt = np.array([target.position(f + offset) for f in frames])
    gt_valid = np.all(np.isfinite(gt), axis=1)
    ok = np.array([t.ok for t in tri])
    tri_points = np.array([t.point for t in tri])
    n_inliers = np.array([len(t.inliers) if t.ok else 0 for t in tri])

    # --- 2D: детекция против проекции точки цели
    per_camera = {}
    for cam in CAMERAS:
        in_view, err_px, rel_err_px = [], [], []
        flags = {"edge_clipped": 0, "too_small": 0, "unreliable": 0}
        for f in frames:
            c = cameras.get(cam, f + offset)
            uv = c.project(gt[f]) if gt_valid[f] and c is not None else np.array([np.nan, np.nan])
            visible = bool(np.all(np.isfinite(uv)) and 0 <= uv[0] < IMAGE_WIDTH and 0 <= uv[1] < IMAGE_HEIGHT)
            in_view.append(visible)
            d = detections[cam].get(f)
            if d is None:
                continue
            for k in flags:
                flags[k] += int(getattr(d, k))
            if visible:
                e = float(np.hypot(d.point_u - uv[0], d.point_v - uv[1]))
                err_px.append(e)
                if not d.unreliable:
                    rel_err_px.append(e)
        in_view = np.array(in_view)
        detected = np.array([f in detections[cam] for f in frames])
        per_camera[cam] = {
            "frames_target_in_view": int(in_view.sum()),
            "frames_detected": int(detected.sum()),
            "detected_and_in_view": int((detected & in_view).sum()),
            "detected_not_in_view": int((detected & ~in_view).sum()),
            "flags": flags,
            "point_vs_target_px_all": stats(err_px),
            "point_vs_target_px_reliable": stats(rel_err_px),
            "triangulation_inlier_frames": int(sum(cam in t.inliers for t in tri if t.ok)),
            "triangulation_outlier_frames": int(sum(cam in t.cams_used and cam not in t.inliers for t in tri if t.ok)),
            "reprojection_px_inliers": stats([t.reprojection_px[cam] for t in tri if t.ok and cam in t.inliers]),
            "_in_view": in_view, "_detected": detected,
        }

    # --- 3D: триангуляция
    tri_err = np.linalg.norm(tri_points - gt, axis=1) / upm
    tri_eval = ok & gt_valid
    by_ncams = {int(k): stats(tri_err[tri_eval & (n_inliers == k)]) for k in (2, 3, 4, 5) if np.any(tri_eval & (n_inliers == k))}

    # --- B-сплайн по точкам триангуляции
    t_ok = frames[ok].astype(float)
    spline = fit_initial_bspline(
        t_ok, tri_points[ok], knot_spacing=args.knot_spacing,
        weights=n_inliers[ok].astype(float), loss="huber", f_scale=args.huber_scale,
    )
    span = (frames >= t_ok[0]) & (frames <= t_ok[-1])
    spline_points = np.full((NUM_VIDEO_FRAMES, 3), np.nan)
    spline_points[span] = spline.evaluate(frames[span].astype(float))
    spl_err = np.linalg.norm(spline_points - gt, axis=1) / upm
    measured = ok
    spline_only = span & ~ok

    # --- разложение ошибки по осям грузовика (по направлению движения цели)
    csv_t = np.clip((frames + offset).astype(float), gt_spline.t_start, gt_spline.t_end)
    gt_vel = gt_spline.derivative().evaluate(csv_t)
    moving = np.linalg.norm(gt_vel, axis=1) > min_speed
    along, cross, vertical = truck_frame_components((spline_points - gt) / upm, gt_vel)
    comp_eval = span & gt_valid & moving
    debiased, bias = debiased_norm(along, cross, vertical, comp_eval)

    # --- кинематика по производным сплайна
    vel = np.full((NUM_VIDEO_FRAMES, 3), np.nan)
    acc = np.full((NUM_VIDEO_FRAMES, 3), np.nan)
    vel[span] = spline.derivative(1).evaluate(frames[span].astype(float)) * args.fps / upm         # м/с
    acc[span] = spline.derivative(2).evaluate(frames[span].astype(float)) * args.fps ** 2 / upm    # м/с²
    speed = np.linalg.norm(vel, axis=1)
    accel = np.linalg.norm(acc, axis=1)
    gt_speed = np.linalg.norm(gt_vel, axis=1) * args.fps / upm
    path_len = float(np.sum(np.linalg.norm(np.diff(spline_points[span], axis=0), axis=1)) / upm)
    gt_path_len = float(np.sum(np.linalg.norm(np.diff(gt_spline.evaluate(csv_t[span]), axis=0), axis=1)) / upm)

    # --- перепроецирование сплайна в каждую камеру против детекций
    spline_reproj = {}
    for cam in CAMERAS:
        errs = []
        for f, d in detections[cam].items():
            if not span[f] or d.unreliable:
                continue
            c = cameras.get(cam, f + offset)
            if c is None:
                continue
            uv = c.project(spline_points[f])
            errs.append(float(np.hypot(uv[0] - d.point_u, uv[1] - d.point_v)))
        spline_reproj[cam] = stats(errs)

    gaps = [(int(a), int(b)) for a, b in find_gaps(ok) if b - a + 1 >= 3]
    metrics = {
        "WARNING": "Модель дообучена на этих же видео: точность детекции оценена на обучающих данных.",
        "weights": str(args.weights),
        "frame_offset": offset,
        "frame_offset_best_by_scan": best_offset,
        "frame_offset_scan": offset_scan,
        "target_csv_issues": target.issues,
        "coverage": {
            "video_frames": NUM_VIDEO_FRAMES,
            "frames_triangulated": int(ok.sum()),
            "frames_by_inlier_cameras": {int(k): int((n_inliers == k).sum()) for k in range(0, 6)},
            "frames_failed_threshold": int(sum(len(t.cams_used) >= 2 and not t.ok for t in tri)),
            "spline_span": [int(t_ok[0]), int(t_ok[-1])],
            "frames_spline_only_in_span": int(spline_only.sum()),
            "gaps_without_measurement_ge3": gaps,
        },
        "triangulation_error_m": stats(tri_err[tri_eval]),
        "triangulation_error_m_by_cameras": by_ncams,
        "spline_error_m_all": stats(spl_err[span & gt_valid]),
        "spline_error_m_measured": stats(spl_err[measured & gt_valid]),
        "spline_error_m_spline_only": stats(spl_err[spline_only & gt_valid]),
        "spline_bias_in_truck_frame_m": bias,
        "spline_error_m_without_bias": stats(debiased[comp_eval]),
        "spline_reprojection_px": spline_reproj,
        "kinematics": {
            "units": {"speed": "м/с", "acceleration": "м/с²", "path": "м"},
            "speed_max": float(np.nanmax(speed)), "speed_median": float(np.nanmedian(speed)),
            "accel_max": float(np.nanmax(accel)),
            "path_length": path_len,
            "target_speed_max": float(np.max(gt_speed[span])), "target_path_length": gt_path_len,
            "speed_error": stats(np.abs(speed - gt_speed)[span]),
        },
        "per_camera": {c: {k: v for k, v in m.items() if not k.startswith("_")} for c, m in per_camera.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    np.savez(out / "trajectory.npz", frames=frames, triangulated=tri_points, triangulated_ok=ok,
             inlier_cameras=n_inliers, spline=spline_points, target=gt, speed=speed, accel=accel,
             knots=spline.knots, control_points=spline.control_points)

    ctx = dict(frames=frames, gt=gt, gt_valid=gt_valid, ok=ok, tri_points=tri_points, tri_err=tri_err,
               n_inliers=n_inliers, span=span, spline_points=spline_points, spl_err=spl_err,
               along=along, cross=cross, vertical=vertical, comp_eval=comp_eval, speed=speed,
               gt_speed=gt_speed, per_camera=per_camera, offset_scan=offset_scan, offset=offset,
               detections=detections, cameras=cameras, upm=upm, tri=tri)
    make_plots(plots, ctx)
    write_report(out, metrics)
    log.info("готово: %s", out / "report.md")
    return {"spline": spline, "ctx": ctx, "metrics": metrics}


# ---------------------------------------------------------------- графики

def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2, "figure.dpi": 110,
    })
    return plt


def make_plots(plots: Path, c: dict) -> None:
    plt = _style()
    upm, frames = c["upm"], c["frames"]
    note = "YOLO (дообучена на этих же видео)"

    # 1. Вид сверху
    fig, ax = plt.subplots(figsize=(11, 6.5))
    gt = c["gt"] / upm
    ax.plot(gt[:, 0], gt[:, 1], color=INK_2, lw=2, label="цель (CSV)")
    tp = c["tri_points"][c["ok"]] / upm
    ax.scatter(tp[:, 0], tp[:, 1], s=8, color=ORANGE, alpha=0.6, lw=0, label="триангуляция")
    sp = c["spline_points"][c["span"]] / upm
    ax.plot(sp[:, 0], sp[:, 1], color=BLUE, lw=2, label="B-сплайн")
    for cam in CAMERAS:
        cobj = c["cameras"].get(cam, 0)
        p = cobj.position / upm
        ax.scatter(*p[:2], marker="^", s=90, color=CAMERA_COLORS[cam], edgecolor=SURFACE, lw=2, zorder=5)
        ax.annotate(f"Cam{cam}", p[:2], xytext=(6, 6), textcoords="offset points", color=INK_2)
    ax.set_aspect("equal")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.set_title(f"Траектория, вид сверху — {note}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots / "01_trajectory_top.png")
    plt.close(fig)

    # 2. Ошибка по кадрам + число камер
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.scatter(frames[c["ok"]], c["tri_err"][c["ok"]], s=6, color=ORANGE, alpha=0.6, lw=0, label="триангуляция")
    a1.plot(frames, c["spl_err"], color=BLUE, lw=2, label="B-сплайн")
    for a, b in find_gaps(c["ok"]):
        a1.axvspan(a - 0.5, b + 0.5, color=GRID, alpha=0.7, lw=0)
    a1.set_ylabel("Ошибка 3D, м")
    a1.set_title(f"Ошибка относительно CSV цели — {note}")
    a1.legend(loc="upper left")
    top = np.nanpercentile(c["tri_err"][c["ok"]], 99) if c["ok"].any() else 1
    a1.set_ylim(0, max(top, np.nanmax(c["spl_err"])) * 1.1)
    a2.step(frames, c["n_inliers"], where="mid", color=INK_2, lw=1.5)
    a2.set_ylabel("Камер")
    a2.set_yticks(range(0, 6))
    a2.set_xlabel("Кадр видео (серым — нет триангуляции, только сплайн)")
    fig.tight_layout()
    fig.savefig(plots / "02_error_vs_frame.png")
    plt.close(fig)

    # 3. Покрытие детекциями
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for i, cam in enumerate(CAMERAS):
        y = len(CAMERAS) - i
        pc = c["per_camera"][cam]
        ax.fill_between(frames, y - 0.4, y + 0.4, where=pc["_in_view"], color=GRID, step="mid", lw=0)
        rel = np.array([f in c["detections"][cam] and not c["detections"][cam][f].unreliable for f in frames])
        unrel = pc["_detected"] & ~rel
        ax.fill_between(frames, y - 0.25, y + 0.25, where=rel, color=BLUE, step="mid", lw=0)
        ax.fill_between(frames, y - 0.25, y + 0.25, where=unrel, color=ORANGE, step="mid", lw=0)
    ax.set_yticks(range(len(CAMERAS), 0, -1), [f"Cam{c_}" for c_ in CAMERAS])
    ax.grid(False)
    ax.set_xlabel("Кадр видео")
    ax.set_title("Покрытие: серый — цель в кадре по CSV, синий — надёжная детекция, оранжевый — с флагом")
    fig.tight_layout()
    fig.savefig(plots / "03_detection_coverage.png")
    plt.close(fig)

    # 4. 2D-ошибка точки объекта против проекции точки цели
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data, labels = [], []
    for cam in CAMERAS:
        e = []
        for f, d in c["detections"][cam].items():
            if d.unreliable or not c["gt_valid"][f]:
                continue
            cobj = c["cameras"].get(cam, f + c["offset"])
            uv = cobj.project(c["gt"][f]) if cobj is not None else np.array([np.nan, np.nan])
            if np.all(np.isfinite(uv)):
                e.append(np.hypot(d.point_u - uv[0], d.point_v - uv[1]))
        data.append(e)
        labels.append(f"Cam{cam}\nn={len(e)}")
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.5)
    for patch, cam in zip(bp["boxes"], CAMERAS):
        patch.set_facecolor(CAMERA_COLORS[cam])
        patch.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_ylabel("Пиксели")
    ax.set_title("Центр маски против проекции точки цели (надёжные детекции)")
    fig.tight_layout()
    fig.savefig(plots / "04_detection_vs_target_px.png")
    plt.close(fig)

    # 5. Ошибка в осях грузовика
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ax, key, name in zip(axes, ("along", "cross", "vertical"),
                             ("вдоль движения", "поперёк движения", "по вертикали")):
        v = np.where(c["comp_eval"], c[key], np.nan)
        ax.plot(frames, v, color=BLUE, lw=1.5)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axhline(np.nanmean(v), color=ORANGE, lw=1.5, ls="--")
        ax.set_ylabel(f"{name}, м")
    axes[0].set_title("Ошибка сплайна в осях грузовика (пунктир — среднее смещение)")
    axes[-1].set_xlabel("Кадр видео")
    fig.tight_layout()
    fig.savefig(plots / "05_error_components.png")
    plt.close(fig)

    # 6. Скорость
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(frames[c["span"]], c["gt_speed"][c["span"]], color=INK_2, label="цель (CSV)")
    a1.plot(frames, c["speed"], color=BLUE, label="B-сплайн")
    a1.set_ylabel("Скорость, м/с")
    a1.set_title(f"Скорость из производной сплайна — {note}")
    a1.legend(loc="upper left")
    a2.plot(frames, c["speed"] - c["gt_speed"], color=BLUE, lw=1.5)
    a2.axhline(0, color=MUTED, lw=1)
    a2.set_ylabel("Разница, м/с")
    a2.set_xlabel("Кадр видео")
    fig.tight_layout()
    fig.savefig(plots / "06_speed.png")
    plt.close(fig)

    # 7. Смещение кадров: сырая ошибка и без постоянного смещения — отдельные панели
    offs = sorted(c["offset_scan"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in zip(axes, ("median_m", "median_debiased_m"),
                              ("Сырая ошибка (смешана с расстоянием кабина ↔ центр)",
                               "Без постоянного смещения в осях грузовика")):
        vals = [c["offset_scan"][o][key] for o in offs]
        best = offs[int(np.argmin(vals))]
        ax.bar(offs, vals, width=0.6, color=[BLUE if o == best else MUTED for o in offs])
        for o, v in zip(offs, vals):
            ax.annotate(f"{v:.2f}", (o, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        color=INK_2, fontsize=9)
        ax.set_xticks(offs)
        ax.set_xlabel("k: кадр CSV = кадр видео + k")
        ax.set_ylabel("Медиана ошибки, м")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Проверка смещения кадров видео ↔ CSV", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(plots / "07_frame_offset_scan.png")
    plt.close(fig)


# ---------------------------------------------------------------- отчёт

def _fmt(s: dict, unit: str = "") -> str:
    if not s or s.get("n", 0) == 0:
        return "нет данных"
    return f"{s['median']:.2f} / {s['p95']:.2f} / {s['rmse']:.2f} / {s['max']:.2f}{unit} (n={s['n']})"


def write_report(out: Path, m: dict) -> None:
    cov = m["coverage"]
    k = m["kinematics"]
    lines = [
        "# Проверка точности на YOLO",
        "",
        f"> ⚠️ {m['WARNING']}",
        "",
        f"Веса: `{m['weights']}`. Смещение кадров: {m['frame_offset']} "
        f"(по скану без постоянного смещения лучшее: {m['frame_offset_best_by_scan']}).",
        "",
        "Испорченные ячейки CSV цели (исключены из оценки): " + ("; ".join(m["target_csv_issues"]) or "нет"),
        "",
        "## Покрытие",
        f"- кадров видео: {cov['video_frames']}, триангулировано: {cov['frames_triangulated']}, "
        f"не прошли порог перепроецирования: {cov['frames_failed_threshold']}",
        "- кадров по числу камер в итоговой точке: "
        + ", ".join(f"{n}: {v}" for n, v in cov["frames_by_inlier_cameras"].items()),
        f"- сплайн на кадрах {cov['spline_span'][0]}–{cov['spline_span'][1]}, из них только сплайном: "
        f"{cov['frames_spline_only_in_span']}",
        "- участки без измерений (≥3 кадров): "
        + (", ".join(f"{a}–{b}" for a, b in cov["gaps_without_measurement_ge3"]) or "нет"),
        "",
        "## Ошибка 3D относительно CSV цели, м (медиана / p95 / RMSE / максимум)",
        f"- триангуляция: {_fmt(m['triangulation_error_m'])}",
    ]
    for n, s in m["triangulation_error_m_by_cameras"].items():
        lines.append(f"  - {n} камеры: {_fmt(s)}")
    b = m["spline_bias_in_truck_frame_m"]
    lines += [
        f"- B-сплайн, все кадры: {_fmt(m['spline_error_m_all'])}",
        f"  - где есть измерение: {_fmt(m['spline_error_m_measured'])}",
        f"  - только сплайн: {_fmt(m['spline_error_m_spline_only'])}",
        f"- среднее смещение в осях грузовика: вдоль {b['along_m']:+.2f}, поперёк {b['cross_m']:+.2f}, "
        f"по вертикали {b['vertical_m']:+.2f} м",
        f"- B-сплайн без постоянного смещения: {_fmt(m['spline_error_m_without_bias'])}",
        "",
        "## Кинематика (из производных сплайна)",
        f"- скорость: медиана {k['speed_median']:.2f} м/с, максимум {k['speed_max']:.2f} м/с "
        f"(цель: максимум {k['target_speed_max']:.2f} м/с)",
        f"- ошибка скорости, м/с: {_fmt(k['speed_error'])}",
        f"- максимум ускорения: {k['accel_max']:.2f} м/с²",
        f"- длина пути: {k['path_length']:.1f} м (цель на том же интервале: {k['target_path_length']:.1f} м)",
        "",
        "## По камерам",
        "| Камера | цель в кадре | детекций | детекция и цель в кадре | детекция без цели в кадре "
        "| флаги край/мала/ненадёжна | центр маски − цель, px | inlier/outlier | перепроец. сплайна, px |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cam, pc in m["per_camera"].items():
        f = pc["flags"]
        s = pc["point_vs_target_px_reliable"]
        r = m["spline_reprojection_px"][cam]
        lines.append(
            f"| Cam{cam} | {pc['frames_target_in_view']} | {pc['frames_detected']} | {pc['detected_and_in_view']} "
            f"| {pc['detected_not_in_view']} | {f['edge_clipped']}/{f['too_small']}/{f['unreliable']} "
            f"| {s.get('median', float('nan')):.1f} (p95 {s.get('p95', float('nan')):.1f}) "
            f"| {pc['triangulation_inlier_frames']}/{pc['triangulation_outlier_frames']} "
            f"| {r.get('median', float('nan')):.1f} (p95 {r.get('p95', float('nan')):.1f}) |"
        )
    lines += ["", "## Графики", ""] + [f"![{p.stem}](plots/{p.name})" for p in sorted((out / "plots").glob("*.png"))]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
