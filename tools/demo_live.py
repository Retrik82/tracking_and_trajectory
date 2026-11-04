"""
Демонстрация трекера реального времени: окно с пятью камерами одновременно, вид сверху и параметры.

По умолчанию — ЖИВАЯ детекция, без подготовки: видео всех камер идут по часам в реальном темпе (30 FPS),
система каждый раз берёт самый свежий синхронный кадр, запускает детектор на пяти камерах, обновляет трекер
и рисует результат. Кадры, которые не успели обработать, пропускаются (как с настоящими камерами): на этом
Mac YOLO на пяти камерах ~80 мс (imgsz 640), классический детектор ~80 мс — обновлений ~10 в секунду.
Трекер получает детекции только текущего кадра (причинно). Эталон CSV рисуется только для сравнения.

    .venv/bin/python tools/demo_live.py                          # живая YOLO + трекер REC
    .venv/bin/python tools/demo_live.py --live classic           # живой детектор без нейросети + REC
    .venv/bin/python tools/demo_live.py --config none            # базовый трекер A
    .venv/bin/python tools/demo_live.py --detections outputs/main/detections   # готовые детекции, каждый кадр

Клавиши: пробел — пауза / продолжить, q или Esc — выход.
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import StreamConfig                       # noqa: E402
from pipeline.streaming import build_components, load_detections_table, load_polygons, load_truck_model  # noqa: E402
from pipeline.variants import VARIANTS                           # noqa: E402
from reconstruction.data_io import load_target, make_camera     # noqa: E402
from reconstruction.fusion import silhouette_points             # noqa: E402

TILE = (640, 360)
SCALE = TILE[0] / 1920
FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
LEVEL_RU = {"multi": "несколько камер", "multi_partial": "несколько камер, часть рёбер закрыта",
            "single": "одна камера", "road": "только дорога/гомография", "predicted": "только прогноз",
            "lost": "потерян", "initializing": "инициализация"}


def font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Reader(threading.Thread):
    """Декодирование видео камеры в отдельном потоке (кадры сразу уменьшаются до плитки)."""

    def __init__(self, path: Path):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(str(path))
        self.q: queue.Queue = queue.Queue(maxsize=8)

    def run(self):
        while True:
            ok, img = self.cap.read()
            self.q.put(cv2.resize(img, TILE, interpolation=cv2.INTER_AREA) if ok else None)
            if not ok:
                return


class Clock:
    """Часы воспроизведения: номер кадра видео, который «идёт» сейчас (с учётом пауз)."""

    def __init__(self, fps: float):
        self.fps = fps
        self.t0 = time.perf_counter()
        self.paused_at = None

    def frame(self) -> float:
        now = self.paused_at if self.paused_at is not None else time.perf_counter()
        return (now - self.t0) * self.fps

    def pause(self):
        self.paused_at = time.perf_counter()

    def resume(self):
        if self.paused_at is not None:
            self.t0 += time.perf_counter() - self.paused_at
            self.paused_at = None


class LiveReader(threading.Thread):
    """Камера «в эфире»: декодирует каждый кадр в его момент времени, хранит последние несколько."""

    def __init__(self, path: Path, clock: Clock, keep: int = 6):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(str(path))
        self.clock, self.keep = clock, keep
        self.frames: dict[int, np.ndarray] = {}
        self.latest = -1
        self.done = False
        self.lock = threading.Lock()

    def run(self):
        idx = 0
        while True:
            while self.clock.frame() < idx:                   # кадр ещё «не снят»
                time.sleep(0.002)
            ok, img = self.cap.read()
            if not ok:
                self.done = True
                return
            with self.lock:
                self.frames[idx] = img
                self.frames.pop(idx - self.keep, None)
                self.latest = idx
            idx += 1


class LiveSource:
    """Самый свежий синхронный кадр всех камер; необработанные кадры пропускаются."""

    def __init__(self, paths: dict[int, Path], fps: float):
        self.clock = Clock(fps)
        self.readers = {c: LiveReader(p, self.clock) for c, p in paths.items()}
        self.last = -1
        self.skipped = 0

    def start(self):
        self.clock = Clock(self.clock.fps)
        for r in self.readers.values():
            r.clock = self.clock
            r.start()

    def next(self):
        while True:
            if any(r.done for r in self.readers.values()):
                return None
            idx = min(r.latest for r in self.readers.values())
            if idx > self.last:
                break
            time.sleep(0.002)
        frames = {}
        for c, r in self.readers.items():
            with r.lock:
                frames[c] = r.frames.get(idx)
        if any(v is None for v in frames.values()):
            self.last = idx
            return self.next()
        self.skipped += max(idx - self.last - 1, 0) if self.last >= 0 else idx
        self.last = idx
        return idx, frames


class FileSource:
    """Все кадры по порядку (для готовых детекций): темп задаёт главный цикл."""

    def __init__(self, paths: dict[int, Path]):
        self.readers = {c: Reader(p) for c, p in paths.items()}
        self.idx = -1
        self.skipped = 0

    def start(self):
        for r in self.readers.values():
            r.start()

    def next(self):
        frames = {c: r.q.get() for c, r in self.readers.items()}
        if any(v is None for v in frames.values()):
            return None
        self.idx += 1
        return self.idx, frames


class YoloLive:
    def __init__(self, weights: Path, imgsz: int, conf: float):
        from detection.yolo import YoloDetector, YoloSettings

        self.d = YoloDetector(YoloSettings(weights_path=weights, imgsz=imgsz, conf=conf))
        self.imgsz, self.conf = imgsz, conf
        self.name = f"YOLO вживую, imgsz {imgsz}"

    def detect(self, idx, frames):
        cams = list(frames)
        res = self.d.model.predict(source=[frames[c] for c in cams], imgsz=self.imgsz, conf=self.conf,
                                   classes=[self.d.class_id], device=self.d.device, retina_masks=True, verbose=False)
        dets, polys = {}, {}
        for c, r in zip(cams, res):
            fr = self.d._convert(r, c, idx)
            if fr.detection is not None:
                dets[c], polys[c] = fr.detection, fr.polygon
        return dets, polys


class ClassicLive:
    def __init__(self, data: Path, cams: list[int]):
        from concurrent.futures import ThreadPoolExecutor

        from detection.classic import ClassicSettings, _State
        from pipeline.detect import camera_rotations

        self.rot = {c: camera_rotations(data, c) for c in cams}
        self.states = {c: _State(ClassicSettings(), self.rot[c][0] if self.rot[c][1] is not None else None)
                       for c in cams}
        self.pool = ThreadPoolExecutor(len(cams))
        self.name = "без нейросети вживую"

    def detect(self, idx, frames):
        def run(c):
            rots = self.rot[c][1]
            R = rots[min(idx, len(rots) - 1)] if rots is not None else None
            return c, self.states[c].step(frames[c], c, idx, R)
        dets, polys = {}, {}
        for c, fr in self.pool.map(run, list(frames)):
            if fr.detection is not None:
                dets[c], polys[c] = fr.detection, fr.polygon
        return dets, polys


class Precomputed:
    def __init__(self, det_dir: Path, cams: list[int]):
        import json

        self.table, _ = load_detections_table(det_dir, cams)
        self.polys = load_polygons(det_dir, cams)
        try:
            classic = str(json.loads((det_dir / "meta.json").read_text()).get("detector", "")).startswith("classic")
        except (OSError, ValueError):
            classic = False
        self.name = ("детектор без нейросети" if classic else "YOLO") + ", детекции посчитаны заранее"

    def detect(self, idx, frames):
        dets = {c: self.table[c][idx] for c in frames if idx in self.table.get(c, {})}
        return dets, {c: self.polys.get((c, idx)) for c in dets}


def project(cam, pts):
    uv = cam.project(np.atleast_2d(pts)) * SCALE
    return uv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("tmp/project2"))
    p.add_argument("--live", choices=["yolo", "classic"], default="yolo", help="детектор, работающий вживую")
    p.add_argument("--detections", type=Path, default=None, help="папка готовых детекций (вместо живой детекции)")
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    p.add_argument("--imgsz", type=int, default=640, help="размер входа YOLO вживую (960 точнее, но вдвое медленнее)")
    p.add_argument("--config", default="configs/upgrade_recommended.json", help="JSON конфигурации или none (вариант A)")
    p.add_argument("--speed", type=float, default=1.0, help="темп воспроизведения (1 — реальный)")
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--start-snapshot", type=int, default=None, help="сохранить кадр с этим номером в PNG и выйти")
    args = p.parse_args(argv)
    import os
    os.chdir(Path(__file__).resolve().parents[1])              # пути по умолчанию — от корня репозитория

    if args.config.lower() == "none":
        cfg = StreamConfig()
        for dotted, value in VARIANTS["A"][1].items():
            s, k = dotted.split(".")
            setattr(getattr(cfg, s), k, value)
        name = "A (базовый)"
    else:
        cfg = StreamConfig.load(Path(args.config))
        name = "REC (курс по силуэту, дорога, неопределённость)"
    cams_ids = cfg.cameras.ids
    model = load_truck_model(cfg)
    tracker, _, _, dims = build_components(cfg, model)
    paths = {c: args.data / cfg.cameras.video_pattern.format(cam=c) for c in cams_ids}
    n = int(cv2.VideoCapture(str(paths[cams_ids[0]])).get(cv2.CAP_PROP_FRAME_COUNT))
    if args.detections is not None:
        detector, source, live = Precomputed(args.detections, cams_ids), FileSource(paths), False
    else:
        print("загрузка детектора…", flush=True)
        detector = YoloLive(args.weights, args.imgsz, cfg.detector.conf) if args.live == "yolo" \
            else ClassicLive(args.data, cams_ids)
        warm = {c: cv2.VideoCapture(str(paths[c])).read()[1] for c in cams_ids}
        if args.live == "yolo":
            detector.detect(0, warm)                         # прогрев (первый вызов модели долгий)
        source, live = LiveSource(paths, cfg.cameras.fps), True
    det_name = detector.name
    try:
        target = load_target(args.data)
    except Exception:                                    # noqa: BLE001
        target = None
    source.start()

    # рамка карты — все камеры и траектория (эталон только задаёт рамку картинки); сверху место под подпись
    pts = [make_camera(args.data, c, 0).position[:2] for c in cams_ids]
    if target is not None:
        pts += list(target.positions[target.valid][:, :2])
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.06 * (hi - lo) + 1500.0                  # запас под значок камеры и подпись
    lo, hi = lo - pad, hi + pad
    MAP_TOP = 28
    MAP = (640, 360 - MAP_TOP)
    k = min(MAP[0] / (hi[0] - lo[0]), MAP[1] / (hi[1] - lo[1]))

    def to_map(xy):
        xy = np.atleast_2d(xy)
        return np.c_[(xy[:, 0] - lo[0]) * k + (MAP[0] - (hi[0] - lo[0]) * k) / 2,
                     MAP_TOP + MAP[1] - ((xy[:, 1] - lo[1]) * k + (MAP[1] - (hi[1] - lo[1]) * k) / 2)]

    f_big, f_mid, f_small = font(28), font(21), font(16)
    trail = deque(maxlen=90)
    err_hist, ax_hist = [], ([], [], [])
    est_path, gt_path = [], []
    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), cfg.cameras.fps, (1920, 960))
    if not args.no_window:
        cv2.namedWindow("Peleng: 5 камер, реальное время", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Peleng: 5 камер, реальное время", 1440, 720)
    period = 1.0 / (cfg.cameras.fps * args.speed)
    t_next = time.perf_counter()
    fps_hist = deque(maxlen=30)
    paused = False
    m = cfg.detector.edge_margin_px
    prev_f = None
    det_ms = 0.0
    while True:
        got = source.next()
        if got is None:
            break
        f, full = got
        frames = {c: cv2.resize(img, TILE, interpolation=cv2.INTER_AREA) for c, img in full.items()}
        cams = {c: make_camera(args.data, c, f + cfg.cameras.frame_offset) for c in cams_ids}
        t_d = time.perf_counter()
        dets, cur_polys = detector.detect(f, full)
        det_ms = (time.perf_counter() - t_d) * 1000
        tuples = []
        for c in cams_ids:
            d = dets.get(c)
            if d is None:
                continue
            valid = np.array([d.x1 > m, d.x2 < cfg.cameras.width - m, d.y1 > m, d.y2 < cfg.cameras.height - m])
            tuples.append((c, np.array([d.x1, d.x2, d.y1, d.y2]), valid, d.confidence, (d.point_u, d.point_v),
                           not d.unreliable, cur_polys.get(c) if cfg.orientation.enabled else None))
        dt_frames = 1.0 if prev_f is None else float(f - prev_f)
        prev_f = f
        t0 = time.perf_counter()
        est = tracker.step(f, {c: cams[c] for c, *_ in tuples}, tuples, dt_frames=dt_frames)
        tick_ms = (time.perf_counter() - t0) * 1000
        ok = np.all(np.isfinite(est.position))
        gt = target.position(f + cfg.cameras.frame_offset) if target is not None else np.full(3, np.nan)
        if ok:
            trail.append(est.position.copy())
            est_path.append(est.position[:2].copy())
        if np.all(np.isfinite(gt)):
            gt_path.append(gt[:2].copy())
        sil = silhouette_points(est.box_center, tracker.heading_dir, model, dims.as_array()) if ok else None

        canvas = np.zeros((960, 1920, 3), np.uint8)
        slots = [(0, 0), (640, 0), (1280, 0), (0, 360), (640, 360)]
        for (x0, y0), c in zip(slots, cams_ids):
            img = frames[c].copy()
            cam = cams[c]
            if c in dets:
                d = dets[c]
                poly = cur_polys.get(c)
                if poly is not None:
                    cv2.polylines(img, [(poly * SCALE).astype(np.int32)], True, (90, 220, 90), 1, cv2.LINE_AA)
                rejected = c in est.rejected_cameras
                col = (60, 60, 230) if rejected else (200, 200, 200)
                cv2.rectangle(img, (int(d.x1 * SCALE), int(d.y1 * SCALE)), (int(d.x2 * SCALE), int(d.y2 * SCALE)),
                              col, 1, cv2.LINE_AA)
            if ok:
                uv = project(cam, sil)
                if np.all(np.isfinite(uv)):
                    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
                    over = img.copy()
                    cv2.fillConvexPoly(over, hull, (230, 140, 40))
                    img = cv2.addWeighted(over, 0.25, img, 0.75, 0)
                    cv2.polylines(img, [hull], True, (255, 170, 60), 2, cv2.LINE_AA)
                tr = project(cam, np.array(trail))
                tr = tr[np.all(np.isfinite(tr), axis=1)].astype(np.int32)
                if len(tr) > 1:
                    cv2.polylines(img, [tr], False, (0, 220, 255), 2, cv2.LINE_AA)
                for pt, colr in ((est.cab_center, (60, 60, 255)), (est.position, (0, 230, 255))):
                    q = project(cam, pt)[0]
                    if np.all(np.isfinite(q)):
                        cv2.circle(img, tuple(q.astype(int)), 5, colr, -1, cv2.LINE_AA)
            if np.all(np.isfinite(gt)):
                q = project(cam, gt)[0]
                if np.all(np.isfinite(q)) and 0 <= q[0] < TILE[0] and 0 <= q[1] < TILE[1]:
                    cv2.drawMarker(img, tuple(q.astype(int)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
            canvas[y0:y0 + 360, x0:x0 + 640] = img
            cv2.rectangle(canvas, (x0, y0), (x0 + 639, y0 + 359), (40, 40, 40), 1)

        # вид сверху
        mp = canvas[360:720, 1280:1920]
        mp[:] = (24, 24, 28)
        for c in cams_ids:
            q = to_map(cams[c].position[:2])[0].astype(int)
            fwd = cams[c].forward[:2] / max(np.linalg.norm(cams[c].forward[:2]), 1e-9)
            cv2.circle(mp, tuple(q), 4, (180, 180, 180), -1)
            cv2.line(mp, tuple(q), tuple((q + fwd * [1, -1] * 18).astype(int)), (180, 180, 180), 2)
            cv2.putText(mp, str(c), tuple(q + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if len(gt_path) > 1:
            cv2.polylines(mp, [to_map(np.array(gt_path)).astype(np.int32)], False, (150, 150, 150), 2, cv2.LINE_AA)
        if len(est_path) > 1:
            cv2.polylines(mp, [to_map(np.array(est_path)).astype(np.int32)], False, (0, 220, 255), 2, cv2.LINE_AA)
        if ok:
            fp = to_map(sil[:, :2])
            cv2.fillConvexPoly(mp, cv2.convexHull(fp.astype(np.float32)).astype(np.int32), (230, 140, 40))
            C = est.reference_covariance[:2, :2]
            if np.all(np.isfinite(C)):
                w, v = np.linalg.eigh(C)
                ax = np.sqrt(np.maximum(w, 0) * 5.991) * k
                ang = -np.degrees(np.arctan2(v[1, 1], v[0, 1]))
                cv2.ellipse(mp, tuple(to_map(est.position[:2])[0].astype(int)), (int(ax[1]), int(ax[0])), ang, 0, 360,
                            (120, 220, 120), 1, cv2.LINE_AA)
            cv2.circle(mp, tuple(to_map(est.position[:2])[0].astype(int)), 4, (0, 230, 255), -1)

        # панель
        now = time.perf_counter()
        fps_hist.append(now)
        fps = (len(fps_hist) - 1) / (fps_hist[-1] - fps_hist[0]) if len(fps_hist) > 2 else float("nan")
        err = float(np.linalg.norm(est.position - gt) / cfg.cameras.units_per_meter) if ok and np.all(np.isfinite(gt)) else float("nan")
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        for (x0, y0), c in zip(slots, cams_ids):
            used = c in dets and c not in est.rejected_cameras
            tag = "детекция" if used else ("отклонена" if c in dets else "нет детекции")
            dr.rectangle([x0 + 6, y0 + 6, x0 + 210, y0 + 32], fill=(0, 0, 0))
            dr.text((x0 + 12, y0 + 9), f"Cam{c} · {tag}", font=f_small,
                    fill=(120, 230, 120) if used else (240, 120, 120) if c in dets else (170, 170, 170))
        px, py = 1280 + 18, 360 + 12
        upm = cfg.cameras.units_per_meter
        ref_r = np.sqrt(np.trace(est.reference_covariance[:2, :2]) * 5.991 / 2) / upm if ok else float("nan")
        W, Y, G, B, D = (230, 230, 230), (255, 220, 120), (120, 220, 120), (160, 200, 255), (170, 170, 170)
        if np.isfinite(err):
            err_hist.append(err)
        eh = np.array(err_hist)
        xyz = lambda v: " / ".join(f"{c:8.2f}" for c in v / upm)  # noqa: E731
        d = (est.position - gt) / upm if ok and np.all(np.isfinite(gt)) else np.full(3, np.nan)
        col1 = [
            (f"Кадр {f} / {n - 1}   t = {f / cfg.cameras.fps:5.2f} с", f_big, (255, 255, 255)),
            (name, f_small, B),
            (f"Наблюдение: {LEVEL_RU.get(est.observation_level, est.observation_level)}", f_mid, W),
            (f"Камер с детекцией: {est.cameras}, в обновлении: {est.cameras_used}", f_mid, W),
            (f"Скорость: {est.speed_mps * 3.6:5.1f} км/ч   курс: {np.degrees(est.heading):6.1f}°" if ok else "Скорость: —",
             f_mid, W),
        ]
        col2 = [
            ("Точка CSV, м:            X        /        Y        /        Z", f_small, D),
            (f"Оценка:   {xyz(est.position)}" if ok else "Оценка: —", f_mid, W),
            (f"Эталон:   {xyz(gt)}" if np.all(np.isfinite(gt)) else "Эталон: —", f_mid, D),
            (f"Ошибка ΔX / ΔY / ΔZ:  {d[0]:+5.2f} / {d[1]:+5.2f} / {d[2]:+5.2f} м" if np.all(np.isfinite(d))
             else "Ошибка: —", f_mid, Y),
            (f"Ошибка 3D сейчас: {err:5.2f} м" if np.isfinite(err) else "Ошибка 3D: —", f_mid, Y),
        ]
        col3 = [
            ("Точность за прогон (ошибка 3D, м):", f_small, D),
            (f"медиана {np.median(eh):4.2f}   p95 {np.percentile(eh, 95):4.2f}   макс {eh.max():4.2f}" if len(eh)
             else "—", f_mid, Y),
            (f"среднее по осям |ΔX| {np.nanmean(np.abs(ax_hist[0])):4.2f}  |ΔY| {np.nanmean(np.abs(ax_hist[1])):4.2f}  "
             f"|ΔZ| {np.nanmean(np.abs(ax_hist[2])):4.2f}" if ax_hist[0] else "", f_small, Y),
            (f"Заявленная неопределённость: 95 % в ~{ref_r:4.1f} м, курс ±{np.degrees(est.heading_std):4.1f}°" if ok
             else "", f_small, G),
            (f"Обновлений: {fps:4.1f} /с   детектор {det_ms:3.0f} мс   трекер {tick_ms:4.1f} мс", f_small, W),
            (f"Детекции: {det_name}; трекер видит только прошлое", f_small, D),
            ("Синий — модель, жёлтый — точка CSV и след, белый крест — эталон", f_small, D),
        ]
        if np.all(np.isfinite(d)):
            for i in range(3):
                ax_hist[i].append(d[i])
        cols = [col1, col2, col3]
        for ci, col_lines in enumerate(cols):
            px, py = 18 + ci * 640, 732
            for text, fnt, col in col_lines:
                dr.text((px, py), text, font=fnt, fill=col)
                py += fnt.size + 12
        dr.text((1290, 366), "Вид сверху: серая — эталон, жёлтая — оценка, эллипс — 95 %", font=f_small, fill=(200, 200, 200))
        canvas = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

        if args.start_snapshot is not None and f >= args.start_snapshot:
            cv2.imwrite("outputs/demo_snapshot.png", canvas)
            break
        if writer is not None:
            writer.write(canvas)
        if not args.no_window:
            cv2.imshow("Peleng: 5 камер, реальное время", canvas)
            if live:
                key = cv2.waitKey(1) & 0xFF
            else:
                t_next += period
                wait = max(1, int((t_next - time.perf_counter()) * 1000))
                key = cv2.waitKey(wait) & 0xFF
                if time.perf_counter() > t_next + 0.5:
                    t_next = time.perf_counter()           # отстали — не догоняем, темп от текущего момента
            if key == ord(" "):
                paused = True
                if live:
                    source.clock.pause()
            while paused:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord(" "), ord("q"), 27):
                    paused = False
                    t_next = time.perf_counter()
                    if live:
                        source.clock.resume()
            if key in (ord("q"), 27):
                break
    if writer is not None:
        writer.release()
        print(f"записано: {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
