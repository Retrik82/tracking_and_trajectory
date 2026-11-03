"""
Потоковый конвейер реального времени (ТЗ 4): кадры → планировщик камер → детекция →
триангуляция и объединение измерений → фильтр Калмана → скользящий B-сплайн → вывод.

Режимы (config.streaming.mode):
  live        источники видео (RTSP/устройства) из config.cameras.live_sources; метки времени — часы захвата;
  replay      записанные видео с исходной частотой кадров (realtime_pacing) и общим стартом — синхронно;
  detections  готовые детекции (<out>/detections/cam{N}.csv) без кадров — проверка алгоритмов и метрик
              без YOLO и видео (единственный режим, который можно прогнать без весов и видеофайлов).

Устройство:
  FrameSource      поток захвата на камеру, ограниченная очередь (новый кадр вытесняет старый);
  Synchronizer     собирает «связку» кадров с метками времени в пределах sync_tolerance; устаревшие
                   кадры (старше max_frame_age) отбрасываются; камера вне допуска в тик не входит;
  CameraScheduler  решает, какие камеры детектировать (ACTIVE / PROBE / SLEEP);
  Detector         YOLO пакетно по активным камерам (или готовые детекции);
  CabTracker       фильтр: рёбра рамок + 3D-точка триангуляции (fusion), dt по меткам времени;
  StreamingBSpline сглаживание в фоновом потоке, SMOOTHED TRACK с задержкой lag_frames;
  StreamRecorder   stream_track.csv, performance_report.json/.md, видео с наложением.

Причинность: на тике t используются только кадры с меткой ≤ t. REALTIME TRACK — состояние фильтра
на t; SMOOTHED TRACK — сплайн на t − lag (помечен своей меткой времени).
"""

import csv
import json
import logging
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detection.schema import Detection
from pipeline.config import StreamConfig, describe
from reconstruction.cab_model import TruckDimensions, truck_axes
from reconstruction.camera import Camera
from reconstruction.camera_scheduler import ACTIVE, CameraScheduler, SchedulerSettings, TrackSnapshot
from reconstruction.data_io import make_camera
from reconstruction.fusion import FusionSettings
from reconstruction.streaming_bspline import StreamingBSpline, StreamingSplineSettings
from reconstruction.truck_model import TruckModel
from reconstruction.visibility import INITIALIZING, MEASURED, PARTIAL, PREDICTED, CabTracker, TrackerSettings

log = logging.getLogger("peleng.stream")


# ---------------------------------------------------------------- кадры и источники

@dataclass
class Frame:
    camera: int
    index: int                       # номер кадра (replay / detections) или порядковый (live)
    timestamp: float                 # с, в общей шкале тика
    image: np.ndarray | None         # BGR; None в режиме detections
    captured_at: float               # time.perf_counter() в момент захвата — для задержки
    detection: Detection | None = None   # готовая детекция (режим detections)


class FrameSource:
    """Поток захвата одной камеры с ограниченной очередью; новый кадр вытесняет старый."""

    def __init__(self, camera: int, queue_size: int = 2, lossless: bool = False):
        self.camera = camera
        self.lossless = lossless                     # файл без темпа: ждать потребителя, а не вытеснять кадры
        self.queue: deque = deque(maxlen=max(1, queue_size))
        self.done = False
        self.dropped = 0
        self.produced = 0
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"capture-cam{self.camera}", daemon=True)
        self._thread.start()

    def _run(self) -> None:       # переопределяется
        self.done = True

    def push(self, frame: Frame) -> None:
        with self._cv:
            while self.lossless and len(self.queue) == self.queue.maxlen and not self._stop.is_set():
                self._cv.wait(0.01)
            if len(self.queue) == self.queue.maxlen:
                self.dropped += 1
            self.queue.append(frame)
            self.produced += 1
            self._cv.notify_all()

    def pop_latest(self, max_age_s: float | None = None) -> Frame | None:
        """Самый новый кадр (старые вытесняются); устаревшие по max_age отбрасываются."""
        with self._cv:
            latest = None
            if self.lossless:
                latest = self.queue.popleft() if self.queue else None
            else:
                while self.queue:
                    f = self.queue.popleft()
                    if max_age_s is not None and f.image is not None and time.perf_counter() - f.captured_at > max_age_s:
                        self.dropped += 1
                        continue
                    latest = f
            self._cv.notify_all()
            return latest

    def peek(self) -> Frame | None:
        with self._cv:
            return self.queue[-1] if self.queue else None

    def wait(self, timeout: float) -> bool:
        with self._cv:
            if self.queue or self.done:
                return True
            return self._cv.wait(timeout)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._cv:
            self.done = True
            self._cv.notify_all()


class VideoSource(FrameSource):
    """Видео (файл — replay с исходным темпом; URL/устройство — live). Общий t0 задаёт синхронный старт."""

    def __init__(self, camera: int, source, fps: float, t0: float, pacing: bool = True, queue_size: int = 2,
                 live: bool = False, max_frames: int | None = None):
        super().__init__(camera, queue_size, lossless=not pacing and not live)
        self.source, self.fps, self.t0, self.pacing, self.live, self.max_frames = source, fps, t0, pacing, live, max_frames

    def _run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            log.error("cam%d: не открыть источник %s", self.camera, self.source)
            with self._cv:
                self.done = True
                self._cv.notify_all()
            return
        index = 0
        while not self._stop.is_set():
            if self.max_frames is not None and index >= self.max_frames:
                break
            if self.pacing and not self.live:
                due = self.t0 + index / self.fps
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            ok, img = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            ts = (now - self.t0) if self.live else index / self.fps
            self.push(Frame(self.camera, index, ts, img, now))
            index += 1
        cap.release()
        with self._cv:
            self.done = True
            self._cv.notify_all()


class DetectionSource(FrameSource):
    """Готовые детекции как «кадры» без изображения (режим detections)."""

    def __init__(self, camera: int, detections: dict[int, Detection], num_frames: int, fps: float, t0: float,
                 pacing: bool = False, queue_size: int = 2, delay_frames: int = 0):
        super().__init__(camera, queue_size, lossless=not pacing)
        self.detections, self.num_frames, self.fps, self.t0, self.pacing = detections, num_frames, fps, t0, pacing
        self.delay_frames = delay_frames          # имитация рассинхронизации: метка времени сдвинута

    def _run(self) -> None:
        for index in range(self.num_frames):
            if self._stop.is_set():
                break
            if self.pacing:
                delay = self.t0 + index / self.fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            self.push(Frame(self.camera, index, (index + self.delay_frames) / self.fps, None, time.perf_counter(),
                            self.detections.get(index)))
        with self._cv:
            self.done = True
            self._cv.notify_all()


# ---------------------------------------------------------------- синхронизация

@dataclass
class Bundle:
    index: int
    timestamp: float
    frames: dict[int, Frame]
    excluded: dict[int, str] = field(default_factory=dict)      # камера -> причина

    @property
    def captured_at(self) -> float:
        return max(f.captured_at for f in self.frames.values()) if self.frames else time.perf_counter()


class Synchronizer:
    """
    Собирает кадры всех камер в связку одного момента времени.

    Опорная метка — минимальная из самых новых меток камер (чтобы ни одна камера не «опережала» на тик
    вперёд): кадр камеры входит в связку, если |t − t_ref| ≤ tolerance; иначе камера исключается с
    причиной (рассинхронизация). Для replay все камеры дают одинаковые метки, и связка — просто кадр N.
    """

    def __init__(self, sources: dict[int, FrameSource], fps: float, tolerance_s: float, max_age_s: float,
                 wait_s: float = 0.5):
        self.sources, self.fps, self.tol, self.max_age, self.wait_s = sources, fps, tolerance_s, max_age_s, wait_s
        self._pending: dict[int, Frame] = {}
        self.excluded_total = 0
        self.last_index = -1

    def all_done(self) -> bool:
        return all(s.done and not s.queue for s in self.sources.values()) and not self._pending

    def next_bundle(self) -> Bundle | None:
        # подтянуть новые кадры
        deadline = time.perf_counter() + self.wait_s
        while True:
            for cam, src in self.sources.items():
                f = src.pop_latest(self.max_age)
                if f is not None:
                    self._pending[cam] = f
            if self._pending:
                break
            if self.all_done():
                return None
            alive = [s for s in self.sources.values() if not s.done]
            if not alive:
                return None
            for s in alive:
                s.wait(0.01)
            if time.perf_counter() > deadline:
                return None
        # дождаться остальных камер в пределах допуска (пока они не done и не отстали)
        t_ref = min(f.timestamp for f in self._pending.values())
        waiting = [c for c, s in self.sources.items() if c not in self._pending and not s.done]
        wait_until = time.perf_counter() + min(self.wait_s, 2.0 / self.fps)
        while waiting and time.perf_counter() < wait_until:
            for cam in list(waiting):
                f = self.sources[cam].pop_latest(self.max_age)
                if f is not None:
                    self._pending[cam] = f
                    waiting.remove(cam)
            if waiting:
                self.sources[waiting[0]].wait(0.002)
        t_ref = min(f.timestamp for f in self._pending.values())
        frames, excluded, keep = {}, {}, {}
        for cam, f in self._pending.items():
            if abs(f.timestamp - t_ref) <= self.tol:
                frames[cam] = f
            else:
                keep[cam] = f                        # кадр из будущего — останется на следующий тик
                excluded[cam] = f"рассинхронизация {abs(f.timestamp - t_ref) * 1000:.0f} мс"
        self._pending = keep
        self.excluded_total += len(excluded)
        index = int(round(t_ref * self.fps))
        if index <= self.last_index:
            index = self.last_index + 1
        self.last_index = index
        return Bundle(index, t_ref, frames, excluded)


# ---------------------------------------------------------------- детекторы

class PrecomputedDetector:
    """Детекции из кадра (режим detections) или из таблицы по (камера, кадр); simulate_ms — имитация стоимости."""

    def __init__(self, table: dict[int, dict[int, Detection]] | None = None, simulate_ms: float = 0.0):
        self.table = table or {}
        self.simulate_ms = simulate_ms
        self.calls = 0
        self.images = 0

    def detect(self, frames: list[Frame]) -> dict[int, Detection | None]:
        self.calls += 1
        self.images += len(frames)
        if self.simulate_ms:
            time.sleep(self.simulate_ms * len(frames) / 1000)
        out = {}
        for f in frames:
            d = f.detection if f.detection is not None else self.table.get(f.camera, {}).get(f.index)
            out[f.camera] = d
        return out


class YoloAdapter:
    """YOLO (ultralytics) пакетно по кадрам активных камер."""

    def __init__(self, cfg):
        from detection.yolo import YoloDetector, YoloSettings

        self.detector = YoloDetector(YoloSettings(weights_path=Path(cfg.weights), imgsz=cfg.imgsz, conf=cfg.conf,
                                                  device=cfg.device, min_mask_area_px=cfg.min_mask_area_px,
                                                  min_reliable_conf=cfg.min_reliable_conf,
                                                  edge_margin_px=cfg.edge_margin_px))
        self.cfg = cfg
        self.calls = 0
        self.images = 0
        self.polygons: dict[tuple[int, int], np.ndarray] = {}

    def detect(self, frames: list[Frame]) -> dict[int, Detection | None]:
        if not frames:
            return {}
        self.calls += 1
        self.images += len(frames)
        d = self.detector
        results = d.model.predict(source=[f.image for f in frames], imgsz=self.cfg.imgsz, conf=self.cfg.conf,
                                  classes=[d.class_id], device=d.device, half=self.cfg.half, retina_masks=True,
                                  verbose=False)
        out = {}
        for f, res in zip(frames, results):
            fr = d._convert(res, f.camera, f.index)
            out[f.camera] = fr.detection
            if fr.polygon is not None:
                self.polygons[(f.camera, f.index)] = fr.polygon
        return out


# ---------------------------------------------------------------- метрики производительности

class StageTimer:
    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    def add(self, name: str, ms: float) -> None:
        self.samples.setdefault(name, []).append(ms)

    def summary(self) -> dict:
        out = {}
        for k, v in self.samples.items():
            a = np.array(v)
            out[k] = {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.median(a)),
                      "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}
        return out


# ---------------------------------------------------------------- запись

@dataclass
class TickRecord:
    index: int
    timestamp: float
    status: str
    quality: str
    box_center: np.ndarray
    reference: np.ndarray
    cab_center: np.ndarray
    velocity_mps: np.ndarray
    speed_mps: float
    accel_mps2: float
    heading_deg: float
    position_std_m: float
    cameras_detected: int
    cameras_used: int
    point_cameras: list
    point_nis: float
    reproj_px: float
    active_cameras: list
    detected_cameras: list
    latency_ms: float
    smoothed_index: float = float("nan")
    smoothed_position: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    smoothed_velocity_mps: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    smoothed_accel_mps2: float = float("nan")
    excluded: dict = field(default_factory=dict)
    rejected_cameras: list = field(default_factory=list)


TRACK_COLUMNS = ["frame", "t_s", "status", "quality", "x", "y", "z", "ref_x", "ref_y", "ref_z", "cab_x", "cab_y", "cab_z",
                 "vx", "vy", "vz", "speed_mps", "accel_mps2", "heading_deg", "position_std_m", "cams_detected",
                 "cams_used", "point_cams", "point_nis", "reproj_px", "active_cams", "detected_cams", "latency_ms",
                 "smoothed_frame", "sx", "sy", "sz", "svx", "svy", "svz", "s_accel_mps2", "excluded", "rejected_cams"]


def record_to_row(r: TickRecord) -> dict:
    return {"frame": r.index, "t_s": f"{r.timestamp:.4f}", "status": r.status, "quality": r.quality,
            "x": r.box_center[0], "y": r.box_center[1], "z": r.box_center[2],
            "ref_x": r.reference[0], "ref_y": r.reference[1], "ref_z": r.reference[2],
            "cab_x": r.cab_center[0], "cab_y": r.cab_center[1], "cab_z": r.cab_center[2],
            "vx": r.velocity_mps[0], "vy": r.velocity_mps[1], "vz": r.velocity_mps[2], "speed_mps": r.speed_mps,
            "accel_mps2": r.accel_mps2, "heading_deg": r.heading_deg, "position_std_m": r.position_std_m,
            "cams_detected": r.cameras_detected, "cams_used": r.cameras_used,
            "point_cams": " ".join(map(str, r.point_cameras)), "point_nis": r.point_nis, "reproj_px": r.reproj_px,
            "active_cams": " ".join(map(str, r.active_cameras)), "detected_cams": " ".join(map(str, r.detected_cameras)),
            "latency_ms": f"{r.latency_ms:.2f}", "smoothed_frame": r.smoothed_index,
            "sx": r.smoothed_position[0], "sy": r.smoothed_position[1], "sz": r.smoothed_position[2],
            "svx": r.smoothed_velocity_mps[0], "svy": r.smoothed_velocity_mps[1], "svz": r.smoothed_velocity_mps[2],
            "s_accel_mps2": r.smoothed_accel_mps2,
            "excluded": "; ".join(f"cam{c}: {why}" for c, why in r.excluded.items()),
            "rejected_cams": " ".join(map(str, r.rejected_cameras))}


# ---------------------------------------------------------------- конвейер

def build_components(cfg: StreamConfig, model: TruckModel | None):
    c = cfg.cameras
    t = cfg.tracker
    dims = TruckDimensions(t.box_length_m * c.units_per_meter, t.box_width_m * c.units_per_meter,
                           t.box_height_m * c.units_per_meter)
    tracker_settings = TrackerSettings(
        fps=c.fps, units_per_meter=c.units_per_meter, motion_model=t.motion_model,
        accel_noise_mps2=t.accel_noise_mps2, lateral_accel_noise_mps2=t.lateral_accel_noise_mps2,
        vertical_accel_noise_mps2=t.vertical_accel_noise_mps2, jerk_noise_mps3=t.jerk_noise_mps3,
        use_edges=t.use_edges, edge_sigma_px=t.edge_sigma_px, huber_k=t.huber_k,
        max_longitudinal_mps2=t.max_longitudinal_mps2, max_lateral_mps2=t.max_lateral_mps2,
        max_correction_per_frame=t.max_correction_per_frame_m * c.units_per_meter,
        heading_rate_limit_dps=t.heading_rate_limit_dps, min_heading_speed_mps=t.heading_min_speed_mps,
        init_frames=t.init_frames, mono_init=t.mono_init, image_width=c.width, image_height=c.height,
        edge_inflation_with_point=cfg.triangulation.edge_inflation_with_point,
    )
    fusion = FusionSettings(mode=cfg.triangulation.mode, sigma_px=cfg.triangulation.sigma_px,
                            max_reproj_px=cfg.triangulation.max_reproj_px,
                            min_ray_angle_deg=cfg.triangulation.min_ray_angle_deg,
                            max_point_std_m=cfg.triangulation.max_point_std_m,
                            covariance_inflation=cfg.triangulation.covariance_inflation,
                            units_per_meter=c.units_per_meter)
    tracker = CabTracker(tracker_settings, dims, model, fusion if cfg.triangulation.mode != "none" else None)
    sc = cfg.scheduler
    scheduler = CameraScheduler(SchedulerSettings(
        enabled=sc.enabled, horizon_frames=sc.horizon_frames, margin_px=sc.margin_px, sigma_scale=sc.sigma_scale,
        active_threshold=sc.active_threshold, probe_threshold=sc.probe_threshold, hysteresis_frames=sc.hysteresis_frames,
        probe_period_frames=sc.probe_period_frames, sleep_period_frames=sc.sleep_period_frames,
        min_active=sc.min_active, max_active=sc.max_active, occluded_patience_frames=sc.occluded_patience_frames,
        force_std_m=sc.force_std_m, force_nis=sc.force_nis, force_reproj_px=sc.force_reproj_px,
        border_px=sc.border_px, prefer_geometry=sc.prefer_geometry, image_width=c.width, image_height=c.height,
        units_per_meter=c.units_per_meter), c.ids, model, dims.as_array())
    sp = cfg.spline
    spline = StreamingBSpline(StreamingSplineSettings(
        window_frames=sp.window_frames, update_every_frames=sp.update_every_frames, lag_frames=sp.lag_frames,
        knot_spacing_frames=sp.knot_spacing_frames, lambda_acceleration=sp.lambda_acceleration,
        lambda_jerk=sp.lambda_jerk, huber_scale=sp.huber_scale_m * c.units_per_meter, max_nfev=sp.max_nfev,
        deadline_ms=sp.deadline_ms, background=sp.background)) if sp.enabled else None
    return tracker, scheduler, spline, dims


def load_truck_model(cfg: StreamConfig) -> TruckModel | None:
    path = Path(cfg.tracker.truck_fbx)
    if path.exists():
        return TruckModel.from_fbx(path, scale=cfg.tracker.truck_scale)
    log.warning("3D-модели %s нет — коробка %.1f × %.2f × %.1f м", path, cfg.tracker.box_length_m,
                cfg.tracker.box_width_m, cfg.tracker.box_height_m)
    return None


class StreamPipeline:
    """Один объект на прогон: источники, компоненты, цикл тиков, запись."""

    def __init__(self, cfg: StreamConfig, data_dir: Path, out_dir: Path | None = None,
                 detections: dict[int, dict[int, Detection]] | None = None, num_frames: int | None = None,
                 model: TruckModel | None = None, pose_provider=None, visualizer=None, detector=None,
                 source_delays: dict[int, int] | None = None):
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir or cfg.output.out_dir)
        self.detections = detections
        self.num_frames = num_frames
        self.model = model if model is not None else load_truck_model(cfg)
        self.tracker, self.scheduler, self.spline, self.dims = build_components(cfg, self.model)
        self.pose_provider = pose_provider or self._pose_from_csv
        self._pose_cache: dict[tuple[int, int], Camera] = {}
        self.visualizer = visualizer
        self.detector = detector
        self.timer = StageTimer()
        self.records: list[TickRecord] = []
        self.sources: dict[int, FrameSource] = {}
        self.source_delays = source_delays or {}
        self.started_at = float("nan")
        self.finished_at = float("nan")
        self.last_estimate = None
        self._prev_t = None

    # --- позы камер
    def _pose_from_csv(self, cam: int, index: int) -> Camera:
        key = (cam, index + self.cfg.cameras.frame_offset)
        if key not in self._pose_cache:
            try:
                self._pose_cache[key] = make_camera(self.data_dir, cam, key[1])
            except ValueError:
                # live: нет строки на этот кадр — последняя известная поза (камеры 1–4 неподвижны)
                last = max((k for k in self._pose_cache if k[0] == cam), default=None)
                if last is None:
                    raise
                self._pose_cache[key] = self._pose_cache[last]
        return self._pose_cache[key]

    # --- источники
    def _make_sources(self) -> None:
        c, s = self.cfg.cameras, self.cfg.streaming
        t0 = time.perf_counter() + 0.2
        for cam in c.ids:
            if s.mode == "detections":
                if self.detections is None:
                    raise ValueError("режим detections требует таблицу детекций")
                self.sources[cam] = DetectionSource(cam, self.detections.get(cam, {}), self.num_frames, c.fps, t0,
                                                    pacing=s.realtime_pacing, queue_size=s.queue_size,
                                                    delay_frames=self.source_delays.get(cam, 0))
            elif s.mode == "replay":
                path = self.data_dir / c.video_pattern.format(cam=cam)
                if not path.exists():
                    log.warning("cam%d: нет видео %s — камера пропущена", cam, path)
                    continue
                self.sources[cam] = VideoSource(cam, str(path), c.fps, t0, s.realtime_pacing, s.queue_size,
                                                live=False, max_frames=s.max_frames)
            elif s.mode == "live":
                src = c.live_sources.get(str(cam)) or c.live_sources.get(cam)
                if src is None:
                    log.warning("cam%d: нет источника в cameras.live_sources — камера пропущена", cam)
                    continue
                self.sources[cam] = VideoSource(cam, src, c.fps, t0, False, s.queue_size, live=True)
            else:
                raise ValueError(f"неизвестный режим {s.mode}")
        if self.detector is None:
            if s.mode == "detections":
                self.detector = PrecomputedDetector(self.detections)
            else:
                self.detector = YoloAdapter(self.cfg.detector)
        for src in self.sources.values():
            src.start()

    # --- один тик
    def tick(self, bundle: Bundle) -> TickRecord:
        cfg = self.cfg
        c = cfg.cameras
        index, t = bundle.index, bundle.timestamp
        tick_start = time.perf_counter()
        cams_pose = {cam: self.pose_provider(cam, index) for cam in bundle.frames}

        # планировщик
        t_s = time.perf_counter()
        snapshot = TrackSnapshot.from_tracker(self.tracker, self.last_estimate)
        plan = self.scheduler.plan(index, snapshot, cams_pose)
        self.timer.add("scheduler", (time.perf_counter() - t_s) * 1000)

        # детекция
        t_s = time.perf_counter()
        to_run = [bundle.frames[cam] for cam in plan.to_run if cam in bundle.frames]
        dets = self.detector.detect(to_run) if to_run else {}
        self.timer.add("detection", (time.perf_counter() - t_s) * 1000)

        # измерения
        tuples, cameras = [], {}
        m = cfg.detector.edge_margin_px
        for cam, d in dets.items():
            if d is None:
                continue
            valid = np.array([d.x1 > m, d.x2 < c.width - m, d.y1 > m, d.y2 < c.height - m])
            tuples.append((cam, np.array([d.x1, d.x2, d.y1, d.y2]), valid, d.confidence,
                           (d.point_u, d.point_v), not d.unreliable))
            cameras[cam] = cams_pose[cam]

        # фильтр
        t_s = time.perf_counter()
        dt_frames = 1.0 if self._prev_t is None else max((t - self._prev_t) * c.fps, 1e-3)
        self._prev_t = t
        est = self.tracker.step(index, cameras, tuples, dt_frames=dt_frames)
        est._timestamp_s = t
        self.last_estimate = est
        self.timer.add("tracker", (time.perf_counter() - t_s) * 1000)
        # обратная связь планировщику: отброшенная гейтом рамка — не «нашла объект»
        self.scheduler.observe(index, {cam: dets.get(cam) is not None and cam not in est.rejected_cameras
                                       for cam in plan.to_run if cam in bundle.frames})

        # сплайн
        smoothed = None
        if self.spline is not None and est.status != INITIALIZING:
            t_s = time.perf_counter()
            if est.status in (MEASURED, PARTIAL):
                self.spline.add(index, est.box_center, est.covariance)
            self.spline.update(index)
            smoothed = self.spline.latest(index)
            self.timer.add("spline_main_thread", (time.perf_counter() - t_s) * 1000)

        latency_ms = (time.perf_counter() - bundle.captured_at) * 1000
        self.timer.add("tick_total", (time.perf_counter() - tick_start) * 1000)
        upm, fps = c.units_per_meter, c.fps
        rec = TickRecord(
            index, t, est.status, est.quality, est.box_center, est.position, est.cab_center, est.velocity_mps,
            est.speed_mps, est.accel_magnitude_mps2, float(np.degrees(est.heading)) if np.isfinite(est.heading) else float("nan"),
            est.position_std / upm if np.isfinite(est.position_std) else float("nan"), est.cameras, est.cameras_used,
            est.point_cameras, est.point_nis, est.reproj_px, plan.active, [cam for cam, d in dets.items() if d is not None],
            latency_ms, excluded=bundle.excluded, rejected_cameras=list(est.rejected_cameras),
        )
        if smoothed is not None:
            rec.smoothed_index = smoothed.frame
            rec.smoothed_position = smoothed.position
            rec.smoothed_velocity_mps = smoothed.velocity * fps / upm
            rec.smoothed_accel_mps2 = float(np.linalg.norm(smoothed.acceleration) * fps ** 2 / upm)
        self.records.append(rec)
        if self.visualizer is not None:
            t_s = time.perf_counter()
            self.visualizer.draw(bundle, rec, est, plan, dets, cams_pose, self)
            self.timer.add("visualization", (time.perf_counter() - t_s) * 1000)
        return rec

    # --- цикл
    def run(self) -> list[TickRecord]:
        cfg, s = self.cfg, self.cfg.streaming
        self.out_dir.mkdir(parents=True, exist_ok=True)
        log.info("поток: %s", describe(cfg))
        self._make_sources()
        if not self.sources:
            raise SystemExit("нет ни одного источника кадров")
        sync = Synchronizer(self.sources, cfg.cameras.fps, s.sync_tolerance_ms / 1000, s.max_frame_age_ms / 1000)
        self.started_at = time.perf_counter()
        ticks = 0
        try:
            while True:
                bundle = sync.next_bundle()
                if bundle is None:
                    if sync.all_done():
                        break
                    continue
                self.tick(bundle)
                ticks += 1
                if s.log_every and ticks % s.log_every == 0:
                    r = self.records[-1]
                    log.info("тик %d t=%.2fс %s активных %s задержка %.1f мс", r.index, r.timestamp, r.status,
                             r.active_cameras, r.latency_ms)
                if s.max_frames is not None and ticks >= s.max_frames:
                    break
        finally:
            self.finished_at = time.perf_counter()
            for src in self.sources.values():
                src.stop()
            if self.spline is not None:
                self.spline.wait_idle()
                self.spline.close()
            if self.visualizer is not None:
                self.visualizer.close()
        self.sync_excluded = sync.excluded_total
        self.write_outputs()
        return self.records

    # --- выход
    def performance(self) -> dict:
        n = len(self.records)
        wall = self.finished_at - self.started_at if n else float("nan")
        lat = np.array([r.latency_ms for r in self.records]) if n else np.array([np.nan])
        active = np.array([len(r.active_cameras) for r in self.records]) if n else np.array([0])
        detected = sum(len(r.detected_cameras) for r in self.records)
        drops = {cam: src.dropped for cam, src in self.sources.items()}
        produced = {cam: src.produced for cam, src in self.sources.items()}
        statuses = {st: int(sum(r.status == st for r in self.records)) for st in (INITIALIZING, MEASURED, PARTIAL, PREDICTED)}
        try:
            import resource

            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:  # noqa: BLE001
            mem_mb = float("nan")
        perf = {
            "mode": self.cfg.streaming.mode, "ticks": n, "wall_s": wall,
            "updates_per_s": n / wall if wall else float("nan"),
            "camera_fps": {cam: produced[cam] / wall if wall else float("nan") for cam in produced},
            "yolo_images_per_s": self.detector.images / wall if wall else float("nan"),
            "yolo_calls": self.detector.calls, "yolo_images": self.detector.images,
            "detections_per_tick": detected / n if n else float("nan"),
            "latency_ms": {"mean": float(np.nanmean(lat)), "p50": float(np.nanmedian(lat)),
                           "p95": float(np.nanpercentile(lat, 95)), "max": float(np.nanmax(lat))},
            "mean_active_cameras": float(active.mean()),
            "dropped_frames": drops, "produced_frames": produced,
            "dropped_fraction": sum(drops.values()) / max(sum(produced.values()), 1),
            "sync_excluded_frames": getattr(self, "sync_excluded", 0),
            "stages": self.timer.summary(),
            "scheduler": self.scheduler.stats(),
            "spline": self.spline.stats() if self.spline is not None else {"fits": 0},
            "statuses": statuses,
            "max_rss_mb": mem_mb,
            "note": ("режим detections: детекция — чтение готовых детекций, время YOLO не измеряется"
                     if self.cfg.streaming.mode == "detections" else "время детекции — реальный инференс YOLO"),
        }
        return perf

    def write_outputs(self) -> None:
        out = self.out_dir
        if self.cfg.output.save_track_csv:
            with open(out / "stream_track.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRACK_COLUMNS)
                w.writeheader()
                for r in self.records:
                    w.writerow(record_to_row(r))
        if self.spline is not None:
            hist = self.spline.history()
            with open(out / "smoothed_track.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "final", "fitted_at"])
                for fr, smp in hist.items():
                    w.writerow([fr, *smp.position, *smp.velocity, *smp.acceleration, int(smp.final), smp.fitted_at])
        if self.cfg.output.performance_report:
            perf = self.performance()
            (out / "performance_report.json").write_text(json.dumps(perf, ensure_ascii=False, indent=2, default=float))
            (out / "performance_report.md").write_text(performance_markdown(perf), encoding="utf-8")
        self.cfg.save(out / "config_used.json")


def performance_markdown(p: dict) -> str:
    lat = p["latency_ms"]
    lines = [
        "# Производительность потокового конвейера", "",
        f"Режим: {p['mode']}. {p['note']}", "",
        f"- тиков: {p['ticks']}, время: {p['wall_s']:.1f} с, выдача 3D-координат: {p['updates_per_s']:.1f} /с",
        f"- задержка кадр → оценка: средняя {lat['mean']:.1f} мс, p50 {lat['p50']:.1f}, p95 {lat['p95']:.1f}, "
        f"макс {lat['max']:.1f} мс",
        f"- детекций (изображений YOLO) в секунду: {p['yolo_images_per_s']:.1f}; на тик: {p['detections_per_tick']:.2f}; "
        f"средне активных камер: {p['mean_active_cameras']:.2f}",
        f"- доля детектированных кадров от захваченных: {p['scheduler'].get('detection_fraction', float('nan')):.2f}",
        f"- пропущено кадров (вытеснено из очередей / устарели): {p['dropped_fraction'] * 100:.1f} %; "
        f"исключено из-за рассинхронизации: {p['sync_excluded_frames']}",
        f"- статусы: {p['statuses']}", f"- память (max RSS): {p['max_rss_mb']:.0f} МБ", "",
        "| этап | n | среднее, мс | p50 | p95 | макс |", "|---|---|---|---|---|---|",
    ]
    for k, v in p["stages"].items():
        lines.append(f"| {k} | {v['n']} | {v['mean_ms']:.2f} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} | {v['max_ms']:.2f} |")
    sp = p.get("spline", {})
    if sp.get("fits"):
        lines += ["", f"Сплайн: {sp['fits']} подгонок, среднее {sp['elapsed_ms_mean']:.2f} мс, p95 {sp['elapsed_ms_p95']:.2f} мс, "
                  f"нарушений бюджета {sp['deadline_violations']} (решатель {sp['solver']})."]
    sc = p.get("scheduler", {})
    if sc:
        lines += ["", "Планировщик камер (доля тиков в состоянии):", "", "| камера | ACTIVE | PROBE | SLEEP |", "|---|---|---|---|"]
        for cam, st in sc.get("state_fraction", {}).items():
            lines.append(f"| {cam} | {st['ACTIVE']:.2f} | {st['PROBE']:.2f} | {st['SLEEP']:.2f} |")
        lines.append(f"\nПринудительных проверок: {sc.get('forced_ticks', 0)} тиков; вызовов детектора на камеру: {sc.get('yolo_calls', 0)}.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- визуализация

class StreamVisualizer:
    """Наложение на кадры выбранных камер (режимы replay/live): рамка, маска, модель, точки, HUD; запись ffmpeg."""

    def __init__(self, cfg: StreamConfig, out_dir: Path, cameras: list[int], target=None):
        import cv2  # noqa: F401 — проверка наличия

        self.cfg, self.out_dir, self.cameras, self.target = cfg, Path(out_dir), cameras, target
        self.writers: dict[int, subprocess.Popen] = {}
        self.trail_rt: deque = deque(maxlen=90)
        self.trail_sm: deque = deque(maxlen=90)
        self.fps_window: deque = deque(maxlen=30)
        self._last_tick = None
        (self.out_dir / "videos").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _bgr(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))

    def _writer(self, cam: int):
        if cam not in self.writers:
            c = self.cfg.cameras
            self.writers[cam] = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{c.width}x{c.height}",
                 "-r", str(c.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                 str(self.out_dir / "videos" / f"stream_cam{cam}.mp4")], stdin=subprocess.PIPE)
        return self.writers[cam]

    def draw(self, bundle: Bundle, rec: TickRecord, est, plan, dets, cams_pose, pipeline: StreamPipeline) -> None:
        import cv2

        now = time.perf_counter()
        if self._last_tick is not None:
            self.fps_window.append(1.0 / max(now - self._last_tick, 1e-6))
        self._last_tick = now
        fps_est = float(np.mean(self.fps_window)) if self.fps_window else float("nan")
        BLUE, YELLOW, AQUA, ORANGE, MAGENTA, MUTED = "#2a78d6", "#eda100", "#1baf7a", "#eb6834", "#e87ba4", "#a3a29c"
        have = np.all(np.isfinite(est.box_center))
        if have:
            self.trail_rt.append(est.cab_center.copy())
        if np.all(np.isfinite(rec.smoothed_position)):
            self.trail_sm.append(rec.smoothed_position.copy())
        upm = self.cfg.cameras.units_per_meter
        for cam in self.cameras:
            f = bundle.frames.get(cam)
            if f is None or f.image is None:
                continue
            img = f.image.copy()
            camera = cams_pose[cam]
            d = dets.get(cam)
            if d is not None:
                cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), self._bgr(MUTED), 1, cv2.LINE_AA)
                poly = getattr(pipeline.detector, "polygons", {}).get((cam, f.index))
                if poly is not None and len(poly) > 2:
                    cv2.polylines(img, [poly.astype(np.int32)], True, self._bgr(MAGENTA), 1, cv2.LINE_AA)
            if have:
                fw, sd, up = (a[0] for a in truck_axes(np.array([[np.cos(est.heading), np.sin(est.heading), 0.0]])))
                if pipeline.model is not None:
                    hull = camera.project(pipeline.model.to_world(est.box_center, fw, sd, up, pipeline.model.hull))
                    hull = hull[np.all(np.isfinite(hull), axis=1)].astype(np.int32)
                    if len(hull) > 2:
                        cv2.polylines(img, [cv2.convexHull(hull)], True, self._bgr(BLUE), 2, cv2.LINE_AA)
                else:
                    from reconstruction.fusion import silhouette_points

                    box = camera.project(silhouette_points(est.box_center, np.array([np.cos(est.heading), np.sin(est.heading), 0]),
                                                           None, pipeline.dims.as_array()))
                    box = box[np.all(np.isfinite(box), axis=1)].astype(np.int32)
                    if len(box) > 2:
                        cv2.polylines(img, [cv2.convexHull(box)], True, self._bgr(BLUE), 2, cv2.LINE_AA)
                for trail, color in ((self.trail_rt, YELLOW), (self.trail_sm, AQUA)):
                    if len(trail) > 1:
                        pts = camera.project(np.array(trail))
                        pts = pts[np.all(np.isfinite(pts), axis=1)].astype(np.int32)
                        if len(pts) > 1:
                            cv2.polylines(img, [pts], False, self._bgr(color), 2, cv2.LINE_AA)
                p = camera.project(est.cab_center)
                if np.all(np.isfinite(p)):
                    marker = cv2.MARKER_CROSS if est.status == PREDICTED else cv2.MARKER_TILTED_CROSS
                    cv2.drawMarker(img, (int(p[0]), int(p[1])), self._bgr(YELLOW if est.status != PREDICTED else ORANGE),
                                   marker, 18, 2, cv2.LINE_AA)
                if np.all(np.isfinite(rec.smoothed_position)):
                    sp = camera.project(rec.smoothed_position)
                    if np.all(np.isfinite(sp)):
                        cv2.circle(img, (int(sp[0]), int(sp[1])), 7, self._bgr(AQUA), 2, cv2.LINE_AA)
            if self.target is not None:
                g = self.target(f.index)
                if g is not None and np.all(np.isfinite(g)):
                    gp = camera.project(g)
                    if np.all(np.isfinite(gp)):
                        cv2.drawMarker(img, (int(gp[0]), int(gp[1])), (60, 60, 220), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
            q = est.box_center / upm if have else np.full(3, np.nan)
            lines = [f"Cam{cam} frame {f.index} t={rec.timestamp:.2f}s  state {plan.states.get(cam, '?')}",
                     f"X {q[0]:.1f} Y {q[1]:.1f} Z {q[2]:.1f} m   std {rec.position_std_m:.2f} m",
                     f"v {rec.speed_mps:.1f} m/s ({rec.speed_mps * 3.6:.0f} km/h)  a {rec.accel_mps2:.1f} m/s2  heading {rec.heading_deg:.0f} deg",
                     f"{rec.status.upper()} / {rec.quality}  cams: active {rec.active_cameras} detected {rec.detected_cameras} tri {rec.point_cameras}",
                     f"smoothed @frame {rec.smoothed_index:.0f}: v {np.linalg.norm(rec.smoothed_velocity_mps):.1f} m/s" if np.isfinite(rec.smoothed_index) else "smoothed: n/a",
                     f"pipeline {fps_est:.1f} ticks/s   latency {rec.latency_ms:.0f} ms"]
            cv2.rectangle(img, (10, 10), (900, 20 + 28 * len(lines)), (20, 20, 20), -1)
            for i, text in enumerate(lines):
                color = self._bgr(ORANGE) if text.startswith(("PREDICTED", "PARTIAL")) else (255, 255, 255)
                cv2.putText(img, text, (20, 36 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
            legend = [("model (Kalman)", BLUE), ("cab center + trail (realtime)", YELLOW), ("smoothed B-spline", AQUA),
                      ("detector box / mask", MUTED), ("target CSV (eval only)", "#dc3c3c")]
            for i, (text, col) in enumerate(legend):
                y = self.cfg.cameras.height - 20 - 26 * (len(legend) - 1 - i)
                cv2.rectangle(img, (16, y - 12), (32, y + 2), self._bgr(col), -1)
                cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            self._writer(cam).stdin.write(img.tobytes())

    def close(self) -> None:
        for w in self.writers.values():
            w.stdin.close()
            w.wait()


# ---------------------------------------------------------------- запуск из командной строки

def load_detections_table(det_dir: Path, cameras: list[int]) -> tuple[dict[int, dict[int, Detection]], int]:
    table, num = {}, 0
    for cam in cameras:
        path = det_dir / f"cam{cam}.csv"
        if not path.exists():
            raise SystemExit(f"нет детекций {path}")
        with open(path, newline="") as f:
            rows = {d.frame: d for d in map(Detection.from_row, csv.DictReader(f))}
        table[cam] = rows
        if rows:
            num = max(num, max(rows) + 1)
    return table, num


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="JSON с параметрами (pipeline/config.py)")
    p.add_argument("--set", action="append", default=[], help="переопределение: раздел.ключ=значение")
    p.add_argument("--mode", choices=["live", "replay", "detections"], default=None)
    p.add_argument("--data", type=Path, default=Path("tmp/project2"))
    p.add_argument("--detections", type=Path, default=None, help="папка с cam{N}.csv (режим detections)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--num-frames", type=int, default=None, help="режим detections: число кадров (по умолчанию из CSV)")
    p.add_argument("--video", action="store_true", help="писать видео с наложением (replay/live)")
    p.add_argument("--no-pacing", action="store_true", help="replay: без ожидания исходного темпа (как можно быстрее)")
    p.add_argument("--print-config", action="store_true")
    args = p.parse_args(argv)
    if args.print_config:
        print(StreamConfig.default_json())
        return
    cfg = StreamConfig.load(args.config, args.set)
    if args.mode:
        cfg.streaming.mode = args.mode
    if args.out:
        cfg.output.out_dir = str(args.out)
    if args.video:
        cfg.output.video = True
    if args.no_pacing:
        cfg.streaming.realtime_pacing = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    detections, num = None, None
    if cfg.streaming.mode == "detections":
        det_dir = args.detections or (Path(cfg.output.out_dir) / "detections")
        detections, num = load_detections_table(det_dir, cfg.cameras.ids)
        num = args.num_frames or num
    visualizer = None
    if cfg.output.video and cfg.streaming.mode != "detections":
        target = None
        try:
            from reconstruction.data_io import load_target

            tt = load_target(args.data)
            target = lambda idx: tt.position(idx + cfg.cameras.frame_offset)  # noqa: E731
        except Exception:  # noqa: BLE001
            pass
        visualizer = StreamVisualizer(cfg, Path(cfg.output.out_dir), cfg.output.video_cameras or cfg.cameras.ids, target)
    pipeline = StreamPipeline(cfg, args.data, Path(cfg.output.out_dir), detections, num, visualizer=visualizer)
    pipeline.run()
    perf = pipeline.performance()
    print(performance_markdown(perf))
    if cfg.output.plots and cfg.output.save_track_csv and pipeline.records:
        from pipeline.stream_plots import plot_stream, read_track

        files = plot_stream(read_track(pipeline.out_dir / "stream_track.csv"), pipeline.out_dir / "plots", args.data,
                            cfg.cameras.fps, cfg.cameras.units_per_meter, cfg.cameras.frame_offset, pipeline.model,
                            pipeline.dims.as_array())
        print("графики: " + ", ".join(str(f) for f in files))
    print(f"результаты: {pipeline.out_dir}")


if __name__ == "__main__":
    main()
