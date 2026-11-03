"""
Сопровождение грузовика в реальном времени с обработкой потери видимости.

Работает строго причинно: оценка на кадре t использует только кадры ≤ t (никакого
заглядывания вперёд), поэтому годится для реального времени.

Почему отдельный модуль. Перед заездом за препятствие рамка детектора несколько кадров
«съёживается» с одной стороны: одно ребро уходит внутрь, противоположное стоит на месте,
уверенность падает (на реальных данных — Cam4, кадры 790–812). Модель, которая верит
такой рамке, принимает это за движение грузовика, отсюда скачки скорости.

Фильтр Калмана с моделью «постоянная скорость» в 3D. Состояние — центр коробки и вектор
скорости. Центр, а не передняя точка: положение центра почти не зависит от поворота
коробки, поэтому ошибка курса не искажает положение и скорость (с передней точкой ошибка
курса сдвигала её вбок, это попадало в скорость, а из неё — снова в курс). Курс не отдельная переменная, а направление
скорости, поэтому коробку нельзя «перевернуть» на 180° (её рамка от этого почти не
меняется, и фильтр с курсом в состоянии так и делал). Измерение — рёбра рамки проекции
коробки на всех камерах этого кадра; они поправляют только положение (итерированное
обновление, Хьюбер), а скорость и курс фильтр выводит из истории поправок положения —
так же, как офлайн-версия берёт курс из касательной сплайна.

Обработка потери видимости:
1. Закрытые рёбра: прогнозная рамка (до обновления) сравнивается с рамкой детектора;
   ребро, ушедшее внутрь при совпадающем противоположном, исключается.
2. Камера, которая снова увидела грузовик, первые кадры входит с пониженным весом
   (известно из прошлого: сколько кадров назад она его увидела).
3. Нет наблюдений — прогноз по вектору скорости, неопределённость растёт, точка не пропадает.
4. Физические пределы после обновления: изменение скорости за кадр ограничено пределами
   продольного и поперечного ускорения, поправка положения за кадр ограничена. Поэтому
   при появлении грузовика скорость не скачет.
"""

from dataclasses import dataclass, field

import numpy as np

from reconstruction.cab_model import (
    CAB_CENTER_BACK,
    CAB_CENTER_UP,
    DEFAULT_DIMENSIONS,
    BoxObservations,
    TruckDimensions,
    box_corners,
    project_box,
    truck_axes,
)
from reconstruction.camera import Camera
from reconstruction.objective import CameraProvider, ObservationGeometry
from reconstruction.triangulation import triangulate_robust

MEASURED, PARTIAL, PREDICTED, INITIALIZING = "measured", "partial", "predicted", "initializing"
STATE_SIZE = 6     # x, y, z, vx, vy, vz (единицы мира и единицы мира за кадр)


@dataclass
class TrackerSettings:
    fps: float = 30.0
    units_per_meter: float = 100.0

    # Шум процесса: СКО ускорения (белый шум ускорения для модели постоянной скорости),
    # отдельно вдоль и поперёк движения. На реальных данных лучше одинаковый (6 м/с²): грузовик
    # в сцене поворачивает резко (поперечное ускорение p90 6.1 м/с²); при 1.5 м/с² медиана
    # ошибки 6.7 м против 5.8 м.
    accel_noise_mps2: float = 6.0             # вдоль движения
    lateral_accel_noise_mps2: float = 6.0     # поперёк движения
    vertical_accel_noise_mps2: float = 1.0

    # Измерения
    # СКО ребра рамки при уверенности 1. Большое, потому что коробка плохо повторяет форму
    # грузовика; из 5 / 10 / 20 px на реальных данных лучше всего 20 (подобрано по CSV цели).
    edge_sigma_px: float = 20.0
    update_iterations: int = 5                # итераций Гаусса–Ньютона в обновлении
    huber_k: float = 2.5                      # порог робастного веса, в СКО невязки
    occlusion_px: float = 12.0                # ребро ушло внутрь прогноза больше чем на…
    occlusion_rel: float = 0.12               # …и больше чем на долю размера рамки
    agree_px: float = 15.0                    # противоположное ребро совпадает с прогнозом
    reappear_ramp_frames: int = 10            # кадров пониженного веса после появления на камере
    reappear_min_weight: float = 0.2
    image_width: int = 1920
    image_height: int = 1080
    edge_margin_px: float = 3.0

    # Физические пределы. Грузовик в этой сцене разгоняется и поворачивает резче реального:
    # по CSV цели продольное ускорение p90 8.8 м/с², поперечное p90 6.1 м/с².
    max_longitudinal_mps2: float = 12.0
    max_lateral_mps2: float = 8.0
    max_correction_per_frame: float = 150.0   # ед. мира: насколько обновление может сдвинуть точку
    min_heading_speed_mps: float = 1.0        # медленнее — курс берётся последний надёжный

    # Инициализация по триангуляции центров масок
    init_frames: int = 8
    init_min_displacement: float = 200.0      # ед. мира: иначе направление не определено

    # Точка вывода
    cab_center_back: float = CAB_CENTER_BACK
    cab_center_up: float = CAB_CENTER_UP

    @property
    def accel_to_units(self) -> float:
        """м/с² -> ед. мира/кадр²."""
        return self.units_per_meter / self.fps ** 2


_F = np.eye(STATE_SIZE)
_F[:3, 3:] = np.eye(3)


def _numeric_jacobian(fun, state: np.ndarray, steps: np.ndarray) -> np.ndarray:
    base = fun(state)
    J = np.zeros((len(base), len(state)))
    for j, h in enumerate(steps):
        if h == 0:
            continue
        d = np.zeros_like(state)
        d[j] = h
        J[:, j] = (fun(state + d) - fun(state - d)) / (2 * h)
    return J


def detect_occluded_edges(predicted: np.ndarray, observed: np.ndarray, edge_valid: np.ndarray,
                          settings: TrackerSettings) -> np.ndarray:
    """
    Рёбра (M, 4) x1, x2, y1, y2, закрытые препятствием: ребро детектора ушло внутрь прогнозной
    рамки, противоположное совпадает, а прогнозное ребро в кадре (иначе это обрезка краем кадра).
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    width = np.abs(predicted[:, 1] - predicted[:, 0])
    height = np.abs(predicted[:, 3] - predicted[:, 2])
    size = np.c_[width, width, height, height]
    threshold = np.maximum(settings.occlusion_px, settings.occlusion_rel * size)
    inward = np.c_[observed[:, 0] - predicted[:, 0], predicted[:, 1] - observed[:, 1],
                   observed[:, 2] - predicted[:, 2], predicted[:, 3] - observed[:, 3]]
    opposite = np.abs(observed - predicted)[:, [1, 0, 3, 2]]
    opposite_ok = (opposite < settings.agree_px) & edge_valid[:, [1, 0, 3, 2]]
    m = settings.edge_margin_px
    in_frame = np.c_[predicted[:, 0] > m, predicted[:, 1] < settings.image_width - m,
                     predicted[:, 2] > m, predicted[:, 3] < settings.image_height - m]
    return edge_valid & in_frame & (inward > threshold) & opposite_ok


@dataclass
class FrameEstimate:
    frame: int
    status: str
    position: np.ndarray                  # опорная точка (низ передней грани по центру)
    box_center: np.ndarray
    cab_center: np.ndarray
    speed_mps: float
    heading: float
    position_std: float                   # СКО положения по горизонтали, ед. мира
    cameras: int                          # камер с детекцией
    cameras_used: int                     # камер, давших ≥3 ребра в обновление
    occluded_edges: int
    frames_since_measurement: int


@dataclass
class TrackResult:
    estimates: list[FrameEstimate] = field(default_factory=list)

    def array(self, name: str) -> np.ndarray:
        return np.array([getattr(e, name) for e in self.estimates])


class CabTracker:
    """
    Причинный трекер: вызывайте step(frame, cameras, detections) на каждом кадре по порядку.

    detections — список кортежей (индекс камеры, рамка x1 x2 y1 y2, рёбра в кадре (4,),
    уверенность, центр маски (u, v) или None, надёжна ли детекция).
    """

    def __init__(self, settings: TrackerSettings | None = None,
                 dimensions: TruckDimensions = DEFAULT_DIMENSIONS):
        self.s = settings or TrackerSettings()
        self.dims = dimensions.as_array()
        self.state: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.heading_dir = np.array([1.0, 0.0, 0.0])
        self._init_buffer: list[tuple[int, np.ndarray]] = []
        self._last_seen: dict[int, int] = {}
        self._appeared: dict[int, int] = {}
        self._last_measured: int | None = None

    def _direction(self, state: np.ndarray) -> np.ndarray:
        """Направление «вперёд» — по скорости; при малой скорости — последнее надёжное."""
        v = state[3:]
        if np.linalg.norm(v[:2]) * self.s.fps / self.s.units_per_meter >= self.s.min_heading_speed_mps:
            self.heading_dir = v / np.linalg.norm(v)
        return self.heading_dir

    def _try_initialize(self, frame: int, cameras: dict[int, Camera], detections) -> bool:
        reliable = [d for d in detections if d[5] and d[4] is not None]
        if len(reliable) >= 2:
            res = triangulate_robust([cameras[d[0]] for d in reliable], [tuple(d[4]) for d in reliable])
            if res.ok:
                self._init_buffer.append((frame, res.point))
        self._init_buffer = [(f, p) for f, p in self._init_buffer if frame - f < 3 * self.s.init_frames]
        if len(self._init_buffer) < self.s.init_frames:
            return False
        frames = np.array([f for f, _ in self._init_buffer], dtype=float)
        points = np.array([p for _, p in self._init_buffer])
        if np.linalg.norm(points[-1, :2] - points[0, :2]) < self.s.init_min_displacement:
            return False
        coef = np.linalg.lstsq(np.c_[frames - frame, np.ones(len(frames))], points, rcond=None)[0]
        velocity, center = coef[0], coef[1]
        self.heading_dir = velocity / np.linalg.norm(velocity)
        # Центр маски ≈ середина коробки
        self.state = np.r_[center, velocity]
        self.P = np.diag([400.0, 400.0, 200.0, 30.0, 30.0, 5.0]) ** 2
        return True

    def _process_noise(self) -> np.ndarray:
        s = self.s
        q = np.array([s.accel_noise_mps2, s.lateral_accel_noise_mps2, s.vertical_accel_noise_mps2]) * s.accel_to_units
        forward = np.r_[self.heading_dir[:2], 0.0]
        forward /= max(np.linalg.norm(forward), 1e-9)
        axes = np.c_[forward, [-forward[1], forward[0], 0.0], [0.0, 0.0, 1.0]]     # вдоль, поперёк, вверх
        Q_acc = axes @ np.diag(q ** 2) @ axes.T
        # Белый шум ускорения за кадр: позиция ~ a/2, скорость ~ a
        G = np.r_[np.eye(3) * 0.5, np.eye(3)]
        return G @ Q_acc @ G.T

    def _reference_point(self, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Опорная точка (низ передней грани) по центру коробки."""
        forward, _, up = truck_axes(direction[None, :])
        return center + self.dims[0] / 2 * forward[0] - self.dims[2] / 2 * up[0]

    def _measure(self, center: np.ndarray, direction: np.ndarray, geometry: ObservationGeometry) -> np.ndarray:
        n = len(geometry.position)
        reference = self._reference_point(center, direction)
        corners = box_corners(np.repeat(reference[None, :], n, axis=0), np.repeat(direction[None, :], n, axis=0),
                              self.dims)
        return project_box(geometry, corners)

    def _apply_limits(self, prior: np.ndarray, posterior: np.ndarray, previous_velocity: np.ndarray) -> np.ndarray:
        s = self.s
        out = posterior.copy()
        dv = out[3:] - previous_velocity
        direction = previous_velocity / max(np.linalg.norm(previous_velocity), 1e-9)
        longitudinal = dv @ direction
        lateral = dv - longitudinal * direction
        long_max = s.max_longitudinal_mps2 * s.accel_to_units
        lat_max = s.max_lateral_mps2 * s.accel_to_units
        longitudinal = np.clip(longitudinal, -long_max, long_max)
        lat_norm = np.linalg.norm(lateral)
        if lat_norm > lat_max:
            lateral = lateral * lat_max / lat_norm
        out[3:] = previous_velocity + longitudinal * direction + lateral
        shift = out[:3] - prior[:3]
        norm = np.linalg.norm(shift)
        if norm > s.max_correction_per_frame:
            out[:3] = prior[:3] + shift * s.max_correction_per_frame / norm
        return out

    def step(self, frame: int, cameras: dict[int, Camera], detections) -> FrameEstimate:
        s = self.s
        for d in detections:
            if self._last_seen.get(d[0], -10 ** 9) < frame - 1:
                self._appeared[d[0]] = frame
            self._last_seen[d[0]] = frame

        if self.state is None:
            if not self._try_initialize(frame, cameras, detections):
                nan3 = np.full(3, np.nan)
                return FrameEstimate(frame, INITIALIZING, nan3, nan3, nan3, np.nan, np.nan, np.nan,
                                     len(detections), 0, 0, -1)
            prior, P_prior = self.state.copy(), self.P.copy()
            previous_velocity = prior[3:].copy()
        else:
            previous_velocity = self.state[3:].copy()
            prior = _F @ self.state
            P_prior = _F @ self.P @ _F.T + self._process_noise()

        state, P = prior, P_prior
        occluded_count, used_cameras = 0, 0
        if detections:
            geometry = ObservationGeometry.from_cameras([cameras[d[0]] for d in detections])
            observed = np.array([d[1] for d in detections], dtype=float)
            valid = np.array([d[2] for d in detections], dtype=bool)
            direction = self._direction(prior)
            predicted = self._measure(prior[:3], direction, geometry)
            occluded = detect_occluded_edges(predicted, observed, valid, s)
            usable = valid & ~occluded & np.isfinite(predicted)
            occluded_count = int(occluded.sum())
            used_cameras = int((usable.sum(axis=1) >= 3).sum())

            if usable.any():
                conf = np.array([max(d[3], 1e-3) for d in detections])
                ramp = np.array([
                    s.reappear_min_weight + (1 - s.reappear_min_weight)
                    * min(1.0, (frame - self._appeared.get(d[0], -10 ** 9)) / max(s.reappear_ramp_frames, 1))
                    for d in detections])
                mask = usable.ravel()
                observed_flat = observed.ravel()[mask]
                R0 = np.repeat(s.edge_sigma_px / np.sqrt(conf * ramp), 4)[mask] ** 2
                P_inv = np.linalg.inv(P_prior)
                measure = lambda x: self._measure(x[:3], direction, geometry).ravel()[mask]  # noqa: E731
                state = prior.copy()
                for _ in range(max(1, s.update_iterations)):
                    Hm = np.zeros((mask.sum(), STATE_SIZE))
                    Hm[:, :3] = _numeric_jacobian(measure, state, np.array([1.0, 1.0, 1.0, 0, 0, 0]))[:, :3]
                    residual = observed_flat - measure(state)
                    S_diag = np.einsum("ij,jk,ik->i", Hm, P_prior, Hm) + R0
                    R = R0 * np.maximum(1.0, np.abs(residual) / np.sqrt(S_diag) / s.huber_k)
                    information = P_inv + Hm.T @ (Hm / R[:, None])
                    gradient = Hm.T @ ((residual + Hm @ (state - prior)) / R)
                    new_state = prior + np.linalg.solve(information, gradient)
                    converged = np.max(np.abs(new_state[:3] - state[:3])) < 1.0
                    state = new_state
                    if converged:
                        break
                P = np.linalg.inv(information)
                P = (P + P.T) / 2
                state = self._apply_limits(prior, state, previous_velocity)

        self.state, self.P = state, P
        status = MEASURED if used_cameras else (PARTIAL if detections else PREDICTED)
        if status == MEASURED:
            self._last_measured = frame
        since = 0 if self._last_measured is None else frame - self._last_measured

        direction = self._direction(state)
        forward, _, up = truck_axes(direction[None, :])
        position = self._reference_point(state[:3], direction)
        cab_center = position - s.cab_center_back * forward[0] + s.cab_center_up * up[0]
        speed = float(np.linalg.norm(state[3:]) * s.fps / s.units_per_meter)
        return FrameEstimate(frame, status, position, state[:3].copy(), cab_center, speed, float(np.arctan2(direction[1], direction[0])),
                             float(np.sqrt(P[0, 0] + P[1, 1])), len(detections), used_cameras, occluded_count, since)


def run_tracker(observations: BoxObservations, camera_provider: CameraProvider, frames,
                points: dict | None = None, reliable: dict | None = None,
                settings: TrackerSettings | None = None,
                dimensions: TruckDimensions = DEFAULT_DIMENSIONS) -> TrackResult:
    """
    Прогон трекера по кадрам по порядку (как в реальном времени).

    points / reliable — {(камера, кадр): (u, v)} и {(камера, кадр): bool} для инициализации
    по триангуляции центров масок.
    """
    tracker = CabTracker(settings, dimensions)
    by_frame: dict[int, list[int]] = {}
    for i, t in enumerate(observations.times.astype(int)):
        by_frame.setdefault(t, []).append(i)
    result = TrackResult()
    for frame in frames:
        idx = by_frame.get(int(frame), [])
        cameras = {int(observations.camera_indices[i]): camera_provider(int(observations.camera_indices[i]),
                                                                         int(observations.frames[i])) for i in idx}
        detections = []
        for i in idx:
            cam = int(observations.camera_indices[i])
            key = (cam, int(frame))
            detections.append((cam, observations.edges[i], observations.edge_valid[i], observations.confidence[i],
                               None if points is None else points.get(key),
                               True if reliable is None else reliable.get(key, False)))
        result.estimates.append(tracker.step(int(frame), cameras, detections))
    return result
