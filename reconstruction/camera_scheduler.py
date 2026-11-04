"""
Адаптивный выбор камер: какие потоки детектировать на этом тике (ТЗ 3.1).

Состояния камеры:
  ACTIVE — YOLO на каждом кадре; камера уверенно видит объект.
  PROBE  — YOLO раз в probe_period_frames; объект может скоро появиться или виден
           геометрически, но не находится (препятствие). Детекция → сразу ACTIVE.
  SLEEP  — YOLO не запускается (кроме редкой контрольной детекции раз в sleep_period_frames);
           объект далеко от поля зрения. Пробуждение — когда прогноз приближает объект к
           полю зрения или растёт неопределённость.

Вероятность видимости. По состоянию фильтра (центр, скорость, ковариация, курс) строятся
сигма-точки положения: центр ± sigma_scale·σ вдоль главных осей ковариации, сейчас и через
horizon_frames (прогноз по скорости). Для каждой точки силуэт модели проецируется в камеру;
доля точек силуэта внутри кадра (с запасом margin_px) и перед камерой, усреднённая по
сигма-точкам, — вероятность видимости p_now и p_horizon.

Геометрическое качество. Камера ближе к объекту даёт больше пикселей на метр; пара камер с
большим углом между лучами лучше определяет глубину. Качество камеры = (1/дальность) ·
(1 + sin угла к лучам уже активных камер). При добирании до min_active и при ограничении
max_active камеры ранжируются по p·качество.

Гистерезис: переходы ACTIVE→PROBE и PROBE→SLEEP требуют, чтобы условие держалось
hysteresis_frames кадров подряд; обратные переходы — немедленно (нельзя терять объект).

Принудительная активация всех камер-кандидатов (p_horizon > probe_threshold): большая
неопределённость положения, большая нормированная невязка (NIS), большая ошибка репроекции,
нет измерений на активных камерах, объект у края кадра активной камеры.

Препятствия: камера, в которой объект геометрически виден, но не найден occluded_patience_frames
кадров подряд, переводится в PROBE (а не остаётся ACTIVE и не считается «не видящей» навсегда):
периодические проверки найдут объект, когда он выедет из-за препятствия.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from reconstruction.camera import Camera
from reconstruction.fusion import silhouette_offsets

ACTIVE, PROBE, SLEEP = "ACTIVE", "PROBE", "SLEEP"


class CameraState(str, Enum):
    ACTIVE = ACTIVE
    PROBE = PROBE
    SLEEP = SLEEP


@dataclass
class SchedulerSettings:
    enabled: bool = True
    horizon_frames: int = 15
    margin_px: float = 40.0
    sigma_scale: float = 2.0
    active_threshold: float = 0.6
    probe_threshold: float = 0.1
    hysteresis_frames: int = 6
    probe_period_frames: int = 10
    sleep_period_frames: int = 90
    min_active: int = 2
    max_active: int | None = None
    occluded_patience_frames: int = 15
    force_std_m: float = 3.0
    force_nis: float = 9.0
    force_reproj_px: float = 60.0
    border_px: float = 80.0
    prefer_geometry: bool = True
    image_width: int = 1920
    image_height: int = 1080
    units_per_meter: float = 100.0


@dataclass
class TrackSnapshot:
    """Что планировщику нужно знать о состоянии трекера на этом тике."""

    center: np.ndarray                       # (3,) ед. мира
    velocity: np.ndarray                     # (3,) ед. мира / кадр
    covariance: np.ndarray                   # (3, 3)
    direction: np.ndarray                    # (3,) курс
    initialized: bool = True
    position_std_m: float = 0.0
    nis: float = float("nan")
    reproj_px: float = float("nan")
    frames_since_measurement: int = 0
    cameras_used: int = 0

    @classmethod
    def from_tracker(cls, tracker, estimate=None) -> "TrackSnapshot":
        if tracker.state is None:
            return cls(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), initialized=False)
        upm = tracker.s.units_per_meter
        return cls(tracker.state[:3].copy(), tracker.state[3:6].copy(), tracker.P[:3, :3].copy(),
                   tracker.heading_dir.copy(), True,
                   float(np.sqrt(tracker.P[0, 0] + tracker.P[1, 1])) / upm,
                   float(estimate.point_nis) if estimate is not None else float("nan"),
                   float(estimate.reproj_px) if estimate is not None else float("nan"),
                   int(estimate.frames_since_measurement) if estimate is not None else 0,
                   int(estimate.cameras_used) if estimate is not None else 0)


@dataclass
class CameraPlan:
    frame: int
    run: dict[int, bool]                     # камера -> запускать ли детекцию на этом тике
    states: dict[int, str]
    p_now: dict[int, float]
    p_horizon: dict[int, float]
    quality: dict[int, float]
    forced: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def active(self) -> list[int]:
        return [c for c, s in self.states.items() if s == ACTIVE]

    @property
    def to_run(self) -> list[int]:
        return [c for c, r in self.run.items() if r]


def visibility_probability(camera: Camera, snapshot: TrackSnapshot, settings: SchedulerSettings,
                           model=None, dims=None, horizon: int = 0,
                           offsets: np.ndarray | None = None) -> tuple[float, float]:
    """
    (вероятность видимости, доля силуэта у края кадра) через horizon кадров.
    Сигма-точки положения: центр ± sigma_scale·σ по главным осям ковариации (7 точек).
    offsets — силуэт относительно центра (silhouette_offsets), чтобы не пересчитывать на каждую камеру.
    """
    if offsets is None:
        offsets = silhouette_offsets(snapshot.direction, model, dims)
    center = snapshot.center + snapshot.velocity * horizon
    cov = snapshot.covariance
    if horizon > 0:
        # неопределённость растёт с горизонтом: грубо, по скорости поворота/ускорения не знаем — доля скорости
        cov = cov + np.outer(snapshot.velocity, snapshot.velocity) * (0.1 * horizon) ** 2
    try:
        w, v = np.linalg.eigh((cov + cov.T) / 2)
    except np.linalg.LinAlgError:
        w, v = np.ones(3), np.eye(3)
    w = np.sqrt(np.maximum(w, 0.0)) * settings.sigma_scale
    samples = np.vstack([center] + [center + s * w[i] * v[:, i] for i in range(3) for s in (-1.0, 1.0)])   # (7, 3)
    W, H, m, b = settings.image_width, settings.image_height, settings.margin_px, settings.border_px
    pts = (samples[:, None, :] + offsets[None, :, :]).reshape(-1, 3)                                     # (7K, 3)
    uv = camera.project(pts)
    front = np.all(np.isfinite(uv), axis=1)
    ok = front & (uv[:, 0] > m) & (uv[:, 0] < W - m) & (uv[:, 1] > m) & (uv[:, 1] < H - m)
    near = front & ((uv[:, 0] < b) | (uv[:, 0] > W - b) | (uv[:, 1] < b) | (uv[:, 1] > H - b))
    return float(ok.mean()), float(near.mean())


def geometric_quality(camera: Camera, center: np.ndarray, others: list[Camera], settings: SchedulerSettings) -> float:
    """Ближе — лучше (пикселей на метр); угол к лучам других активных камер — лучше глубина."""
    ray = center - camera.position
    dist_m = max(np.linalg.norm(ray), 1.0) / settings.units_per_meter
    q = 100.0 / dist_m
    if settings.prefer_geometry and others:
        r = ray / np.linalg.norm(ray)
        sines = []
        for o in others:
            ro = center - o.position
            ro /= max(np.linalg.norm(ro), 1e-9)
            sines.append(np.sqrt(max(1.0 - float(r @ ro) ** 2, 0.0)))
        q *= 1.0 + float(np.max(sines))
    return float(q)


class CameraScheduler:
    def __init__(self, settings: SchedulerSettings, camera_ids, model=None, dims=None):
        self.s = settings
        self.ids = [int(c) for c in camera_ids]
        self.model, self.dims = model, dims
        self.states: dict[int, str] = {c: ACTIVE for c in self.ids}
        self._below_active: dict[int, int] = {c: 0 for c in self.ids}
        self._below_probe: dict[int, int] = {c: 0 for c in self.ids}
        self._last_run: dict[int, int] = {c: -10 ** 9 for c in self.ids}
        self._last_detected: dict[int, int] = {c: -10 ** 9 for c in self.ids}
        self._not_found: dict[int, int] = {c: 0 for c in self.ids}     # видна геометрически, но не найдена, кадров подряд
        self.yolo_calls = 0
        self.history: list[CameraPlan] = []

    # ---------------------------------------------------------------- обратная связь

    def observe(self, frame: int, detected: dict[int, bool]) -> None:
        """После детекции: какие из запущенных камер нашли объект."""
        for cam, found in detected.items():
            self._last_run[cam] = frame
            self.yolo_calls += 1
            if found:
                self._last_detected[cam] = frame
                self._not_found[cam] = 0
                if self.states[cam] != ACTIVE:
                    self.states[cam] = ACTIVE                          # ТЗ: детекция в PROBE → ACTIVE
                    self._below_active[cam] = 0
            else:
                self._not_found[cam] += 1

    # ---------------------------------------------------------------- план

    def plan(self, frame: int, snapshot: TrackSnapshot | None, cameras: dict[int, Camera]) -> CameraPlan:
        s = self.s
        ids = [c for c in self.ids if c in cameras]
        if not s.enabled or snapshot is None or not snapshot.initialized:
            for c in ids:
                self.states[c] = ACTIVE
            plan = CameraPlan(frame, {c: True for c in ids}, dict(self.states), {c: 1.0 for c in ids},
                              {c: 1.0 for c in ids}, {c: 1.0 for c in ids}, False,
                              ["планировщик выключен или трек не инициализирован: все камеры ACTIVE"])
            self.history.append(plan)
            return plan

        p_now, p_hor, border = {}, {}, {}
        offsets = silhouette_offsets(snapshot.direction, self.model, self.dims)
        for c in ids:
            p_now[c], border[c] = visibility_probability(cameras[c], snapshot, s, self.model, self.dims, 0, offsets)
            p_hor[c], _ = visibility_probability(cameras[c], snapshot, s, self.model, self.dims, s.horizon_frames, offsets)
        reasons = []

        # --- переходы с гистерезисом
        for c in ids:
            st = self.states[c]
            if st == ACTIVE:
                self._below_active[c] = self._below_active[c] + 1 if p_now[c] < s.active_threshold else 0
                occluded = p_now[c] >= s.active_threshold and self._not_found[c] >= s.occluded_patience_frames
                if self._below_active[c] >= s.hysteresis_frames or occluded:
                    self.states[c] = PROBE
                    self._below_active[c] = 0
                    reasons.append(f"cam{c}: ACTIVE→PROBE ({'не находит, хотя видна' if occluded else 'уходит из кадра'})")
            elif st == PROBE:
                self._below_probe[c] = self._below_probe[c] + 1 if p_hor[c] < s.probe_threshold else 0
                if self._below_probe[c] >= s.hysteresis_frames:
                    self.states[c] = SLEEP
                    self._below_probe[c] = 0
                    reasons.append(f"cam{c}: PROBE→SLEEP")
                elif p_now[c] >= s.active_threshold and frame - self._last_detected[c] <= s.probe_period_frames:
                    self.states[c] = ACTIVE
                    reasons.append(f"cam{c}: PROBE→ACTIVE (видна и недавно найдена)")
            else:  # SLEEP
                if p_hor[c] >= s.probe_threshold:
                    self.states[c] = PROBE
                    self._below_probe[c] = 0
                    reasons.append(f"cam{c}: SLEEP→PROBE (объект приближается к полю зрения)")

        # --- принудительная активация
        forced = []
        if snapshot.position_std_m > s.force_std_m:
            forced.append(f"СКО положения {snapshot.position_std_m:.1f} м")
        if np.isfinite(snapshot.nis) and snapshot.nis > s.force_nis:
            forced.append(f"NIS {snapshot.nis:.1f}")
        if np.isfinite(snapshot.reproj_px) and snapshot.reproj_px > s.force_reproj_px:
            forced.append(f"репроекция {snapshot.reproj_px:.0f} px")
        if snapshot.frames_since_measurement > 0 or snapshot.cameras_used == 0:
            forced.append("нет измерений на активных камерах")
        if any(border[c] > 0.3 for c in ids if self.states[c] == ACTIVE):
            forced.append("объект у края кадра активной камеры")
        if forced:
            for c in ids:
                if p_hor[c] >= s.probe_threshold and self.states[c] != ACTIVE:
                    self.states[c] = PROBE
            reasons.append("принудительная проверка кандидатов: " + "; ".join(forced))

        # --- геометрическое качество и минимум активных
        active = [c for c in ids if self.states[c] == ACTIVE]
        quality = {c: geometric_quality(cameras[c], snapshot.center, [cameras[o] for o in active if o != c], s)
                   for c in ids}
        occluded_now = {c for c in ids if self._not_found[c] >= s.occluded_patience_frames}
        candidates = sorted((c for c in ids if self.states[c] != ACTIVE and p_now[c] >= s.probe_threshold
                             and c not in occluded_now),
                            key=lambda c: -p_now[c] * quality[c])
        while len(active) < s.min_active and candidates:
            c = candidates.pop(0)
            self.states[c] = ACTIVE
            active.append(c)
            reasons.append(f"cam{c}: добрана до min_active")
        if s.max_active is not None and len(active) > s.max_active:
            for c in sorted(active, key=lambda c: p_now[c] * quality[c])[: len(active) - s.max_active]:
                self.states[c] = PROBE
                reasons.append(f"cam{c}: ACTIVE→PROBE (max_active)")

        # --- кого детектировать на этом тике
        run = {}
        for c in ids:
            st = self.states[c]
            since = frame - self._last_run[c]
            if st == ACTIVE:
                run[c] = True
            elif st == PROBE:
                run[c] = since >= s.probe_period_frames or bool(forced)
            else:
                run[c] = s.sleep_period_frames > 0 and since >= s.sleep_period_frames
        plan = CameraPlan(frame, run, dict(self.states), p_now, p_hor, quality, bool(forced), reasons)
        self.history.append(plan)
        return plan

    # ---------------------------------------------------------------- статистика

    def stats(self) -> dict:
        if not self.history:
            return {}
        n = len(self.history)
        states = np.array([[p.states.get(c, SLEEP) for c in self.ids] for p in self.history])
        runs = np.array([[p.run.get(c, False) for c in self.ids] for p in self.history])
        return {
            "ticks": n,
            "mean_active": float((states == ACTIVE).sum(axis=1).mean()),
            "mean_detections_per_tick": float(runs.sum(axis=1).mean()),
            "detection_fraction": float(runs.mean()),
            "forced_ticks": int(sum(p.forced for p in self.history)),
            "state_fraction": {c: {st: float((states[:, i] == st).mean()) for st in (ACTIVE, PROBE, SLEEP)}
                               for i, c in enumerate(self.ids)},
            "yolo_calls": int(self.yolo_calls),
        }
