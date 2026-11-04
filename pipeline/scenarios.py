"""
Синтетические сценарии для проверки доработок (детерминированные, без видео и без YOLO).

Сцена: дорога z = 0, четыре неподвижные камеры по сторонам на высоте 12–18 м, грузовик-коробка
16.5 × 2.55 × 4 м (как коробка трека без FBX). Траектория задаётся профилем скорости и скорости поворота;
курс кузова = направление скорости центра. На каждый кадр и камеру — рамка проекции коробки с шумом,
контур маски (выпуклая оболочка проекции с шумом вершин), центроид маски.

Сценарии (ТЗ 13): прямо, медленный поворот, резкий поворот, частичное перекрытие, нет двух камер,
последовательная потеря стерео (одна камера 4 с), полная потеря и восстановление, ложные детекции,
ошибки маски, систематическое смещение рамки, неверная ориентация маски, несовпадение модели.

Метрики по точке вывода (низ передней грани): медиана / p95 / макс ошибки, ошибка курса, доля ошибок
в 95 %-эллипсе заявленной ковариации, |e|/σ, число инициализаций, кадров «потерян», скачок положения
при восстановлении. CSV цели здесь нет — истина известна из генератора.

  python -m pipeline.scenarios --variants A,ALL --out outputs/upgrade/scenarios
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from detection.schema import Detection
from pipeline.config import StreamConfig
from pipeline.streaming import StreamPipeline
from pipeline.variants import VARIANTS
from reconstruction.cab_model import box_corners
from reconstruction.camera import Camera

FPS = 30.0
UPM = 100.0
DIMS = np.array([1650.0, 255.0, 400.0])           # длина, ширина, высота коробки, ед. мира


def look_at(position, target, f=1600.0) -> Camera:
    d = np.asarray(target, float) - np.asarray(position, float)
    return Camera(position, 0.0, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]),
                  f, f, 960, 540, 1920, 1080)


@dataclass
class Scenario:
    name: str
    frames: int = 300
    speed_mps: float = 10.0
    yaw_dps: callable = None                     # yaw_dps(frame) -> °/с
    speed_fn: callable = None                    # speed_fn(frame) -> м/с (иначе постоянная)
    drop: callable = None                        # drop(cam, frame) -> True — детекции нет
    shrink: callable = None                      # shrink(cam, frame) -> px: правое ребро уходит внутрь (перекрытие)
    false_det: callable = None                   # false_det(cam, frame, rng) -> рамка x1 y1 x2 y2 или None
    neighbour: callable = None                   # neighbour(cam, frame) -> True: вместо грузовика рамка соседнего объекта
    mask_error: callable = None                  # mask_error(cam, frame, poly, rng) -> испорченный контур
    bbox_bias: dict = field(default_factory=dict)  # {cam: (dx1, dy1, dx2, dy2)} px
    mask_heading_error_deg: callable = None      # (cam, frame) -> ошибка курса силуэта маски, °
    render_scale: tuple = (1.0, 1.0, 1.0)        # размеры отрисованного грузовика относительно модели трека
    noise_px: float = 1.5
    description: str = ""


def trajectory(sc: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Центр коробки (N, 3) и курс (N,) по профилю скорости и скорости поворота; старт в (−40 м, 0)."""
    pos = np.zeros((sc.frames, 3))
    head = np.zeros(sc.frames)
    p = np.array([-4000.0, 0.0, DIMS[2] / 2])
    h = 0.0
    for f in range(sc.frames):
        pos[f], head[f] = p, h
        v = (sc.speed_fn(f) if sc.speed_fn else sc.speed_mps) * UPM / FPS
        h += math.radians(sc.yaw_dps(f) if sc.yaw_dps else 0.0) / FPS
        p = p + v * np.array([math.cos(h), math.sin(h), 0.0])
    return pos, head


def cameras_for(pos: np.ndarray) -> list[Camera]:
    c = pos.mean(axis=0)
    c[2] = 0.0
    span = max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), 4000.0)
    r = span * 0.6 + 4000.0
    return [look_at(c + [-r, -r * 0.8, 1500.0], c), look_at(c + [r, -r * 0.7, 1800.0], c),
            look_at(c + [r * 0.6, r, 1200.0], c), look_at(c + [-r * 0.9, r * 0.7, 1600.0], c)]


def reference_point(center: np.ndarray, heading: float) -> np.ndarray:
    """Низ передней грани коробки (точка вывода трека без FBX)."""
    f = np.array([math.cos(heading), math.sin(heading), 0.0])
    return center + DIMS[0] / 2 * f - np.array([0.0, 0.0, DIMS[2] / 2])


def render(sc: Scenario, seed: int = 0):
    """Таблица детекций {cam: {frame: Detection}}, контуры масок, камеры и истина."""
    rng = np.random.default_rng(seed)
    pos, head = trajectory(sc)
    cams = cameras_for(pos)
    dims = DIMS * np.array(sc.render_scale)
    table = {i: {} for i in range(len(cams))}
    polys = {}
    truth_ref = np.zeros((sc.frames, 3))
    for f in range(sc.frames):
        d = np.array([math.cos(head[f]), math.sin(head[f]), 0.0])
        ref = pos[f] + dims[0] / 2 * d - np.array([0.0, 0.0, dims[2] / 2])
        ref[2] = 0.0
        truth_ref[f] = ref                                   # низ передней грани настоящего (отрисованного) грузовика
        corners = box_corners(ref[None, :], d[None, :], dims)[0]
        for i, cam in enumerate(cams):
            if sc.drop is not None and sc.drop(i, f):
                continue
            uv = cam.project(corners)
            if not np.all(np.isfinite(uv)):
                continue
            x1, x2 = uv[:, 0].min(), uv[:, 0].max()
            y1, y2 = uv[:, 1].min(), uv[:, 1].max()
            if x2 < 0 or x1 > 1920 or y2 < 0 or y1 > 1080:
                continue
            poly_src = uv
            if sc.mask_heading_error_deg is not None:
                err = math.radians(sc.mask_heading_error_deg(i, f))
                if err:
                    dd = np.array([math.cos(head[f] + err), math.sin(head[f] + err), 0.0])
                    ref2 = pos[f] + dims[0] / 2 * dd - np.array([0.0, 0.0, dims[2] / 2])
                    ref2[2] = 0.0
                    poly_src = cam.project(box_corners(ref2[None, :], dd[None, :], dims)[0])
            e = np.array([x1, y1, x2, y2]) + rng.normal(0, sc.noise_px, 4)
            if sc.shrink is not None:
                e[2] -= sc.shrink(i, f)
            if i in sc.bbox_bias:
                e += np.asarray(sc.bbox_bias[i], float)
            e = np.clip(e, 0, [1919, 1079, 1919, 1079])
            vertices = ConvexHull(poly_src).vertices
            hull = poly_src[vertices] + rng.normal(0, sc.noise_px, (len(vertices), 2))
            if sc.shrink is not None and sc.shrink(i, f) > 0:
                hull[:, 0] = np.minimum(hull[:, 0], e[2])
            if sc.mask_error is not None:
                hull = sc.mask_error(i, f, hull, rng)
            c = hull.mean(axis=0)
            table[i][f] = Detection(i, f, 0.9, float(e[0]), float(e[1]), float(e[2]), float(e[3]),
                                    float(c[0]), float(c[1]), "mask_centroid", 5000.0, float(c[0]), float(c[1]))
            polys[(i, f)] = hull.astype(np.float32)
            if sc.neighbour is not None and sc.neighbour(i, f):
                # соседний объект: такая же рамка, сдвинутая на 1.1 ширины вбок (IoU ≈ 0, близко к прогнозу)
                w = e[2] - e[0]
                nb = e + np.array([1.1 * w, 0.0, 1.1 * w, 0.0])
                cx, cy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
                table[i][f] = Detection(i, f, 0.8, *map(float, nb), float(cx), float(cy), "mask_centroid", 4000.0,
                                        float(cx), float(cy))
                polys[(i, f)] = (hull + np.array([1.1 * w, 0.0])).astype(np.float32)
            if sc.false_det is not None:
                fb = sc.false_det(i, f, rng)
                if fb is not None:
                    # ложная детекция заменяет настоящую (детектор выдаёт один объект на кадр)
                    cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                    table[i][f] = Detection(i, f, 0.7, *map(float, fb), float(cx), float(cy), "mask_centroid",
                                            3000.0, float(cx), float(cy))
                    polys[(i, f)] = np.array([[fb[0], fb[1]], [fb[2], fb[1]], [fb[2], fb[3]], [fb[0], fb[3]]],
                                             np.float32)
    return table, polys, cams, truth_ref, head


def scenario_config(variant: str) -> StreamConfig:
    cfg = StreamConfig()
    cfg.cameras.ids = [0, 1, 2, 3]
    cfg.streaming.mode = "detections"
    cfg.streaming.realtime_pacing = False
    cfg.streaming.log_every = 0
    cfg.spline.background = False
    cfg.output.performance_report = False
    cfg.output.save_track_csv = False
    cfg.tracker.truck_fbx = "models/NO_SUCH_MODEL.fbx"         # синтетика — коробка
    cfg.tracker.edge_sigma_px = 4.0
    for dotted, value in VARIANTS[variant][1].items():
        section, key = dotted.split(".")
        setattr(getattr(cfg, section), key, value)
    return cfg


def run_scenario(sc: Scenario, variant: str, out_dir: Path | None = None, seed: int = 0) -> dict:
    table, polys, cams, truth, head = render(sc, seed)
    cfg = scenario_config(variant)
    pipe = StreamPipeline(cfg, Path("."), out_dir or Path("/tmp/peleng_scenarios"), table, sc.frames, model=None,
                          pose_provider=lambda cam, idx: cams[cam], polygons=polys)
    pipe.out_dir.mkdir(parents=True, exist_ok=True)
    records = pipe.run()
    err, head_err, d2, ratio, levels = [], [], [], [], {}
    est = np.full((sc.frames, 3), np.nan)
    for r in records:
        levels[r.observation_level] = levels.get(r.observation_level, 0) + 1
        if not np.all(np.isfinite(r.reference)):
            continue
        est[r.index] = r.reference
        e = (r.reference - truth[r.index]) / UPM
        err.append(float(np.linalg.norm(e)))
        head_err.append(abs((r.heading_deg - math.degrees(head[r.index]) + 180) % 360 - 180))
        C = r.reference_cov_m2[:2, :2]
        if np.all(np.isfinite(C)) and np.linalg.det(C) > 0:
            d2.append(float(e[:2] @ np.linalg.solve(C, e[:2])))
            ratio.append(float(np.linalg.norm(e[:2]) / math.sqrt(np.trace(C))))
    err, head_err, d2 = np.array(err), np.array(head_err), np.array(d2)
    step = np.linalg.norm(np.diff(est, axis=0), axis=1) / UPM
    expected = np.linalg.norm(np.diff(truth, axis=0), axis=1) / UPM
    jump = np.nanmax(np.abs(step - expected)) if np.any(np.isfinite(step)) else float("nan")
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")  # noqa: E731
    return {
        "frames_with_estimate": int(len(err)),
        "error_m": {"median": q(err, 50), "p95": q(err, 95), "max": float(err.max()) if len(err) else float("nan")},
        "heading_error_deg": {"median": q(head_err, 50), "p95": q(head_err, 95)},
        "coverage_95": float(np.mean(d2 <= 5.991)) if len(d2) else float("nan"),
        "error_to_sigma_median": q(np.array(ratio), 50),
        "initializations": pipe.tracker.initializations,
        "lost_frames": levels.get("lost", 0),
        "levels": levels,
        "max_position_jump_m": float(jump),
        "rejected_detections": int(sum(len(r.rejected_cameras) for r in records)),
        "tick_ms_p95": float(np.percentile([r.latency_ms for r in records], 95)) if records else float("nan"),
    }


def _yaw_window(a, b, rate):
    return lambda f: rate if a <= f < b else 0.0


def scenarios() -> list[Scenario]:
    def false_det(cam, f, rng):
        if cam == 1 and (rng.random() < 0.1 or 150 <= f < 165):
            x, y = rng.uniform(100, 1600), rng.uniform(100, 800)
            return (x, y, x + rng.uniform(80, 300), y + rng.uniform(60, 200))
        return None

    def mask_err(cam, f, poly, rng):
        if rng.random() < 0.25:
            c = poly.mean(axis=0)
            if rng.random() < 0.5:
                return c + (poly - c) * np.array([0.5, 1.0])         # половина маски по ширине
            return poly + rng.normal(0, 25, poly.shape)               # рваный контур
        return poly

    return [
        Scenario("прямо", 240, 12.0, description="движение без изменения курса"),
        Scenario("медленный поворот", 360, 2.0, _yaw_window(90, 330, 15.0),
                 speed_fn=lambda f: 8.0 if f < 40 else max(2.0, 8.0 - 0.3 * (f - 40)),
                 description="разгон 8 м/с, затем 2 м/с и поворот 60° за 8 с (курс по скорости шумит)"),
        Scenario("резкий поворот", 240, 10.0, _yaw_window(80, 140, 45.0),
                 description="10 м/с, 90° за 2 с"),
        Scenario("частичное перекрытие", 240, 10.0, _yaw_window(60, 180, 10.0),
                 shrink=lambda c, f: 0.0 if not (c == 0 and 90 <= f < 140) else 6.0 * (f - 90),
                 description="камера 0: правое ребро уходит за препятствие на 50 кадров"),
        Scenario("нет двух камер", 300, 10.0, _yaw_window(100, 200, 12.0),
                 drop=lambda c, f: c in (0, 1) and 100 <= f < 200,
                 description="камеры 0 и 1 пропадают на 100 кадров"),
        Scenario("потеря стерео", 360, 8.0, _yaw_window(120, 240, 10.0),
                 drop=lambda c, f: c != 2 and 100 <= f < 220,
                 description="только камера 2 на 120 кадров (4 с)"),
        Scenario("потеря и восстановление", 300, 10.0, _yaw_window(100, 160, 15.0),
                 drop=lambda c, f: 120 <= f < 150,
                 description="нет ни одной камеры 30 кадров, затем все возвращаются"),
        Scenario("ложные детекции", 300, 10.0, _yaw_window(80, 200, 10.0), false_det=false_det,
                 description="камера 1: 10 % ложных рамок + серия 15 кадров"),
        Scenario("чужой объект при появлении", 300, 10.0, _yaw_window(80, 200, 8.0),
                 drop=lambda c, f: c == 2 and 100 <= f < 150,
                 neighbour=lambda c, f: c == 2 and 150 <= f < 175,
                 description="камера 2 пропадает на 50 кадров и первые 25 кадров после возвращения видит соседний объект"),
        Scenario("ошибки маски", 300, 6.0, _yaw_window(80, 220, 12.0), mask_error=mask_err,
                 description="25 % масок испорчены (половина ширины или рваный контур)"),
        Scenario("смещение рамки", 240, 10.0, _yaw_window(60, 180, 10.0), bbox_bias={0: (0, 0, 0, 15), 2: (-10, 0, 0, 0)},
                 description="систематическое смещение рамок двух камер на 10–15 px"),
        Scenario("неверная ориентация маски", 300, 6.0, _yaw_window(80, 220, 12.0),
                 mask_heading_error_deg=lambda c, f: 40.0 if c == 3 else 0.0,
                 description="маски камеры 3 нарисованы с ошибкой курса 40°"),
        Scenario("несовпадение модели", 240, 10.0, _yaw_window(60, 180, 10.0), render_scale=(1.12, 1.0, 1.1),
                 description="настоящий грузовик на 12 % длиннее и на 10 % выше модели"),
    ]


def run_all(variants: list[str], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for sc in scenarios():
        results[sc.name] = {"description": sc.description,
                            **{v: run_scenario(sc, v, out / "runs") for v in variants}}
    lines = ["# Синтетические сценарии", "",
             "Детерминированная синтетика (коробка 16.5 × 2.55 × 4 м, 4 камеры, шум рамки 1.5 px). Истина — из генератора. "
             "Точка — низ передней грани. «в 95 %» — доля ошибок внутри 95 %-эллипса заявленной ковариации (идеал 95 %).", "",
             "| сценарий | вариант | ошибка медиана / p95 / макс, м | курс медиана / p95, ° | в 95 % | |e|/σ | "
             "инициализаций | «потерян» | макс. скачок, м | отклонено детекций |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, d=2: "—" if not np.isfinite(x) else f"{x:.{d}f}"  # noqa: E731
    for name, res in results.items():
        for v in variants:
            r = res[v]
            lines.append(f"| {name} | {v} | {f(r['error_m']['median'])} / {f(r['error_m']['p95'])} / {f(r['error_m']['max'])} | "
                         f"{f(r['heading_error_deg']['median'], 1)} / {f(r['heading_error_deg']['p95'], 1)} | "
                         f"{r['coverage_95'] * 100:.0f} % | {f(r['error_to_sigma_median'])} | {r['initializations']} | "
                         f"{r['lost_frames']} | {f(r['max_position_jump_m'])} | {r['rejected_detections']} |")
    lines += ["", "Описание сценариев:", ""] + [f"- **{n}** — {r['description']}" for n, r in results.items()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "scenarios.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float))
    return results


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", default="A,ALL")
    p.add_argument("--out", type=Path, default=Path("outputs/upgrade/scenarios"))
    args = p.parse_args(argv)
    run_all(args.variants.split(","), args.out)
    print((args.out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
