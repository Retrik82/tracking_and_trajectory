"""
Сопровождение грузовика в реальном времени с обработкой потери видимости.

Работает строго причинно: оценка на кадре t использует только кадры ≤ t (никакого
заглядывания вперёд), поэтому годится для реального времени.

Почему отдельный модуль. Перед заездом за препятствие рамка детектора несколько кадров
«съёживается» с одной стороны: одно ребро уходит внутрь, противоположное стоит на месте,
уверенность падает (на реальных данных — Cam4, кадры 790–812). Модель, которая верит
такой рамке, принимает это за движение грузовика, отсюда скачки скорости.

Фильтр Калмана в 3D. Состояние — центр модели грузовика, скорость и (в модели «постоянное
ускорение») ускорение. Центр, а не передняя точка: положение центра почти не зависит от
поворота, поэтому ошибка курса не искажает положение и скорость. Курс — отдельная переменная
с собственной динамикой: цель — направление скорости, но изменение ограничено скоростью
поворота, при малой скорости курс держится (см. _update_heading). Рамка одна и та же при
повороте на 180°, поэтому курс нельзя измерять по рамке — отсюда известные ограничения.

Измерения (объединяются в одном итерированном обновлении, информационная форма):
1. Рёбра рамки проекции модели на каждой камере (пиксели; поправляют только положение).
2. 3D-точка триангуляции центроидов (reconstruction/fusion.py) с её ковариацией
   (измерение центра модели напрямую). Если точка и рёбра одной камеры получены из одной
   маски, они коррелированы: ковариация рёбер этой камеры увеличивается
   (edge_inflation_with_point). Точка проходит гейт по нормированной невязке (NIS).

Обработка потери видимости:
1. Закрытые рёбра: прогнозная рамка (до обновления) сравнивается с рамкой детектора;
   ребро, ушедшее внутрь при совпадающем противоположном, исключается.
2. Камера, которая снова увидела грузовик, первые кадры входит с пониженным весом.
3. Нет наблюдений — прогноз по модели движения, неопределённость растёт, точка не пропадает.
4. Физические пределы после обновления: изменение скорости за кадр ограничено пределами
   продольного и поперечного ускорения, поправка положения за кадр ограничена.

Шаг времени dt (в кадрах) может быть любым: матрицы перехода и шума пересчитываются.
"""

from dataclasses import dataclass, field

import numpy as np

from reconstruction.cab_model import (
    CAB_CENTER_BACK,
    CAB_CENTER_UP,
    DEFAULT_DIMENSIONS,
    BoxObservations,
    TruckDimensions,
    truck_axes,
)
from reconstruction.camera import Camera
from reconstruction.fusion import FusionSettings, PointMeasurement, build_point_measurement, silhouette_offsets
from reconstruction.objective import CameraProvider, ObservationGeometry
from reconstruction.triangulation import triangulate_robust
from reconstruction.truck_model import TruckModel

MEASURED, PARTIAL, PREDICTED, INITIALIZING = "measured", "partial", "predicted", "initializing"
STATE_SIZE = 6     # модель cv: x, y, z, vx, vy, vz (единицы мира и единицы мира за кадр); ca — 9


@dataclass
class TrackerSettings:
    fps: float = 30.0
    units_per_meter: float = 100.0
    motion_model: str = "cv"                  # cv — постоянная скорость, ca — постоянное ускорение

    # Шум процесса: СКО ускорения (белый шум ускорения для модели постоянной скорости),
    # отдельно вдоль и поперёк движения. На реальных данных лучше одинаковый (6 м/с²): грузовик
    # в сцене поворачивает резко (поперечное ускорение p90 6.1 м/с²); при 1.5 м/с² медиана
    # ошибки 6.7 м против 5.8 м.
    accel_noise_mps2: float = 6.0             # вдоль движения
    lateral_accel_noise_mps2: float = 6.0     # поперёк движения
    vertical_accel_noise_mps2: float = 1.0
    jerk_noise_mps3: float = 20.0             # модель ca: СКО рывка

    # Измерения
    # СКО ребра рамки при уверенности 1. Большое, потому что коробка плохо повторяет форму
    # грузовика; из 5 / 10 / 20 px на реальных данных лучше всего 20 (подобрано по CSV цели).
    use_edges: bool = True
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
    point_gate_nis: float = 16.0              # 3D-точка с NIS больше — выброс, не используется
    edge_inflation_with_point: float = 2.0    # рёбра камеры, вошедшей в точку, коррелированы с ней: R × k
    # Гейт детекций по согласованности с прогнозом (ложные рамки): рамка, у которой IoU с прогнозной
    # меньше gate_iou и центр дальше gate_distance размеров прогнозной рамки, отбрасывается — но только
    # пока трек «уверен» (измерение было не позже gate_max_age кадров и СКО меньше gate_max_std_m)
    gate_iou: float = 0.05
    gate_distance: float = 1.5
    gate_max_age: int = 30
    gate_max_std_m: float = 5.0
    # Контроль расходимости: если триангуляция ≥2 камер столько кадров подряд не проходит гейт NIS
    # (или ≥2 рамки отброшены гейтом), трек считается ушедшим и переинициализируется по стереоточкам
    reinit_after_bad_frames: int = 10
    # Инициализация по одной камере: дальность из высоты рамки и известной высоты модели. По умолчанию
    # выключена: одна камера не даёт глубины (на синтетике ошибка десятки метров), а ТЗ требует не выдавать
    # недостоверные координаты. Включайте для установок, где объект подолгу видит одна камера; такие
    # оценки помечены quality = mono до подтверждения второй камерой
    mono_init: bool = False

    # Физические пределы. Грузовик в этой сцене разгоняется и поворачивает резче реального:
    # по CSV цели продольное ускорение p90 8.8 м/с², поперечное p90 6.1 м/с².
    max_longitudinal_mps2: float = 12.0
    max_lateral_mps2: float = 8.0
    max_correction_per_frame: float = 150.0   # ед. мира: насколько обновление может сдвинуть точку
    limit_relax_std_m: float = 1e9            # при СКО положения больше — предел сдвига растёт (по умолчанию выключено:
                                              # на синтетике с ложными рамками ослабление давало рывки до 60 м)
    min_heading_speed_mps: float = 1.0        # медленнее — курс держится
    heading_rate_limit_dps: float = 60.0      # предел скорости поворота курса, °/с
    min_turn_radius_m: float = 8.0            # курс не быстрее v / R_min (Аккерман): при 3 м/с ≤ 21°/с, при 1 м/с ≤ 7°/с
    heading_velocity_smoothing: float = 0.5   # цель курса — направление сглаженной скорости (EMA с этим весом)

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

    @property
    def state_size(self) -> int:
        return 9 if self.motion_model == "ca" else 6


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


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU рамок (M, 4) в формате x1 x2 y1 y2."""
    ix = np.maximum(0.0, np.minimum(a[:, 1], b[:, 1]) - np.maximum(a[:, 0], b[:, 0]))
    iy = np.maximum(0.0, np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 2], b[:, 2]))
    inter = ix * iy
    area_a = np.maximum(a[:, 1] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 2], 0)
    area_b = np.maximum(b[:, 1] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 2], 0)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def gate_detections(predicted: np.ndarray, observed: np.ndarray, settings: TrackerSettings) -> np.ndarray:
    """True — рамка несогласована с прогнозом (ложная детекция или другой объект)."""
    predicted = np.asarray(predicted, float)
    observed = np.asarray(observed, float)
    iou = box_iou(predicted, observed)
    size = np.maximum(np.maximum(predicted[:, 1] - predicted[:, 0], predicted[:, 3] - predicted[:, 2]), 1.0)
    pc = np.c_[(predicted[:, 0] + predicted[:, 1]) / 2, (predicted[:, 2] + predicted[:, 3]) / 2]
    oc = np.c_[(observed[:, 0] + observed[:, 1]) / 2, (observed[:, 2] + observed[:, 3]) / 2]
    dist = np.linalg.norm(pc - oc, axis=1) / size
    return (iou < settings.gate_iou) & (dist > settings.gate_distance) & np.all(np.isfinite(predicted), axis=1)


@dataclass
class FrameEstimate:
    frame: int
    status: str
    position: np.ndarray                  # точка CSV цели (с моделью) или низ передней грани (с коробкой)
    box_center: np.ndarray
    cab_center: np.ndarray
    speed_mps: float
    heading: float                        # курс, радианы (ось X → Y)
    position_std: float                   # СКО положения по горизонтали, ед. мира
    cameras: int                          # камер с детекцией
    cameras_used: int                     # камер, давших ≥3 ребра в обновление
    occluded_edges: int
    frames_since_measurement: int
    # --- расширенный выход (ТЗ 3.3)
    rejected_cameras: list = field(default_factory=list)   # детекции, отброшенные гейтом как несогласованные
    velocity_mps: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))      # Vx, Vy, Vz, м/с
    acceleration_mps2: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))  # м/с²
    accel_magnitude_mps2: float = float("nan")
    covariance: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan))    # ковариация центра, ед.²
    point_used: bool = False              # 3D-точка триангуляции вошла в обновление
    point_cameras: list = field(default_factory=list)
    point_nis: float = float("nan")       # нормированная невязка точки (χ² с 3 степенями свободы)
    point_std_m: float = float("nan")     # СКО триангулированной точки, м
    reproj_px: float = float("nan")       # средняя невязка рёбер после обновления, px
    reproj_by_camera: dict = field(default_factory=dict)
    dt_frames: float = 1.0
    quality: str = "none"                 # none | predicted | edges | point | point+edges

    @property
    def timestamp_s(self) -> float:       # заполняется потоковым конвейером при наличии меток времени
        return getattr(self, "_timestamp_s", float("nan"))


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
                 dimensions: TruckDimensions = DEFAULT_DIMENSIONS, model: TruckModel | None = None,
                 fusion: FusionSettings | None = None):
        """
        model — форма грузовика из 3D-модели: рамка измерения считается по проекции её
        выпуклой оболочки, а точки вывода (точка CSV, центр кабины) берутся из геометрии модели.
        Без модели — коробка dimensions и центр кабины по cab_center_back / cab_center_up.
        fusion — настройки триангуляции центроидов как 3D-измерения (None — только рёбра).
        """
        self.s = settings or TrackerSettings()
        self.model = model
        self.fusion = fusion
        self.dims = model.dimensions.copy() if model is not None else dimensions.as_array()
        self.n = self.s.state_size
        self.state: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.heading = 0.0                                  # курс, радианы
        self.heading_dir = np.array([1.0, 0.0, 0.0])
        self._init_buffer: list[tuple[int, np.ndarray]] = []
        self._mono_buffer: list[tuple[int, np.ndarray]] = []
        self._confirmed = False                             # трек подтверждён ≥2 камерами (гейт рамок включается)
        self._v_smooth: np.ndarray | None = None            # сглаженная скорость для курса
        self._last_seen: dict[int, int] = {}
        self._appeared: dict[int, int] = {}
        self._last_measured: int | None = None
        self._last_confirmed: int | None = None             # последний кадр с подтверждением ≥2 камерами
        self._bad_streak = 0                                # кадров подряд, когда стерео не согласуется с треком
        self._last_frame: int | None = None
        self._prev_velocity: np.ndarray | None = None
        self._accel_est = np.zeros(3)                       # сглаженное ускорение (модель cv), ед./кадр²
        self._offsets_cache: tuple = (None, None)

    # ---------------------------------------------------------------- модель движения

    def _transition(self, dt: float) -> np.ndarray:
        F = np.eye(self.n)
        F[:3, 3:6] = np.eye(3) * dt
        if self.n == 9:
            F[:3, 6:9] = np.eye(3) * dt ** 2 / 2
            F[3:6, 6:9] = np.eye(3) * dt
        return F

    def _axes(self) -> np.ndarray:
        forward = np.r_[self.heading_dir[:2], 0.0]
        forward /= max(np.linalg.norm(forward), 1e-9)
        return np.c_[forward, [-forward[1], forward[0], 0.0], [0.0, 0.0, 1.0]]     # вдоль, поперёк, вверх

    def _process_noise(self, dt: float) -> np.ndarray:
        s = self.s
        axes = self._axes()
        if self.n == 9:
            q = np.full(3, s.jerk_noise_mps3 * s.units_per_meter / s.fps ** 3)
            Q_j = axes @ np.diag(q ** 2) @ axes.T
            G = np.r_[np.eye(3) * dt ** 3 / 6, np.eye(3) * dt ** 2 / 2, np.eye(3) * dt]
            return G @ Q_j @ G.T
        q = np.array([s.accel_noise_mps2, s.lateral_accel_noise_mps2, s.vertical_accel_noise_mps2]) * s.accel_to_units
        Q_acc = axes @ np.diag(q ** 2) @ axes.T
        # Белый шум ускорения за dt: позиция ~ a·dt²/2, скорость ~ a·dt
        G = np.r_[np.eye(3) * dt ** 2 / 2, np.eye(3) * dt]
        return G @ Q_acc @ G.T

    # ---------------------------------------------------------------- курс

    def _speed_mps(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(state[3:5]) * self.s.fps / self.s.units_per_meter)

    def _update_heading(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Курс как отдельная переменная: цель — направление скорости, но поворот не быстрее
        heading_rate_limit_dps, а при скорости ниже порога курс держится. Так курс не
        прыгает от шума скорости на медленных участках и при остановке.
        """
        v = state[3:6]
        a = self.s.heading_velocity_smoothing
        self._v_smooth = v.copy() if self._v_smooth is None else (1 - a) * self._v_smooth + a * v
        speed = self._speed_mps(state)
        if speed >= self.s.min_heading_speed_mps:
            target = float(np.arctan2(self._v_smooth[1], self._v_smooth[0]))
            # физический предел: ω ≤ v / R_min — при малой скорости курс почти не меняется,
            # и шум направления медленной скорости не раскачивает ориентацию
            rate = min(np.radians(self.s.heading_rate_limit_dps), speed / max(self.s.min_turn_radius_m, 0.1))
            limit = rate / self.s.fps * max(dt, 1e-6)
            delta = _wrap(target - self.heading)
            self.heading = _wrap(self.heading + float(np.clip(delta, -limit, limit)))
            horizontal = np.hypot(v[0], v[1])
            slope = v[2] / max(horizontal, 1e-9)
        else:
            slope = 0.0
        d = np.array([np.cos(self.heading), np.sin(self.heading), slope])
        self.heading_dir = d / np.linalg.norm(d)
        return self.heading_dir

    def _direction(self, state: np.ndarray) -> np.ndarray:
        """Текущее направление «вперёд» (без изменения курса)."""
        return self.heading_dir

    # ---------------------------------------------------------------- инициализация

    def _mono_point(self, camera: Camera, detection) -> np.ndarray | None:
        """
        Точка по одной камере: дальность из высоты рамки и известной высоты модели
        (z_cam ≈ f_y · H / h_px), направление — луч через центроид маски. Грубо (ошибка дальности
        ~ ошибка высоты рамки в долях), но позволяет начать трек, когда объект видит одна камера.
        """
        edges, valid, centroid = detection[1], detection[2], detection[4]
        if centroid is None or not (valid[2] and valid[3]):
            return None
        h_px = float(edges[3] - edges[2])
        if h_px < 8:
            return None
        # вертикальный размер рамки — не только высота H: при наклоне камеры в него входит и глубина
        # силуэта (ближний нижний и дальний верхний углы): от W (вид сбоку) до L (вид спереди). Курс
        # неизвестен — берём W: дальность не завышается, для вида спереди занижается (до ~30 %)
        pitch = abs(float(np.arcsin(np.clip(camera.forward[2], -1.0, 1.0))))
        effective_height = float(self.dims[2]) * np.cos(pitch) + float(self.dims[1]) * np.sin(pitch)
        depth = camera.f_y * effective_height / h_px
        from reconstruction.triangulation import pixel_to_ray

        ray = pixel_to_ray(camera, float(centroid[0]), float(centroid[1]))
        along = float(ray @ camera.forward)
        if along <= 1e-6:
            return None
        return camera.position + ray * depth / along

    def _try_initialize(self, frame: int, cameras: dict[int, Camera], detections, allow_mono: bool = True) -> bool:
        """
        Инициализация по буферу точек: триангуляция ≥2 камер (точно) или, если за всё окно буфера
        ни одной стереоточки не было, точки по одной камере (грубо: дальность по высоте рамки, скорость
        неизвестна — ноль с большой дисперсией; трек неподтверждён, гейт рамок выключен). Пока трек
        не подтверждён, стереоточки продолжают копиться, и как только их хватает, состояние
        ПЕРЕинициализируется по ним (allow_mono=False): стерео всегда важнее догадки по одной камере.
        """
        reliable = [d for d in detections if d[5] and d[4] is not None]
        if len(reliable) >= 2:
            res = triangulate_robust([cameras[d[0]] for d in reliable], [tuple(d[4]) for d in reliable])
            if res.ok:
                self._init_buffer.append((frame, res.point))
        elif len(reliable) == 1 and self.s.mono_init and allow_mono:
            p = self._mono_point(cameras[reliable[0][0]], reliable[0])
            if p is not None:
                self._mono_buffer.append((frame, p))
        window = 3 * self.s.init_frames
        self._init_buffer = [(f, p) for f, p in self._init_buffer if frame - f < window]
        self._mono_buffer = [(f, p) for f, p in self._mono_buffer if frame - f < window]
        mono = False
        buffer = self._init_buffer
        if len(buffer) < self.s.init_frames:
            if not allow_mono or self._init_buffer or len(self._mono_buffer) < 2 * self.s.init_frames:
                return False
            buffer, mono = self._mono_buffer, True
        frames = np.array([f for f, _ in buffer], dtype=float)
        points = np.array([p for _, p in buffer])
        if not mono and np.linalg.norm(points[-1, :2] - points[0, :2]) < self.s.init_min_displacement:
            return False
        if mono:
            # дальность по высоте рамки шумит (десятки метров): положение — медиана буфера, скорость
            # неизвестна (ноль с большой дисперсией); курс — вдоль горизонтальной оси луча камеры
            center = np.median(points, axis=0)
            velocity = np.zeros(3)
            cam = cameras[reliable[0][0]] if reliable else None
            fwd = cam.forward if cam is not None else np.array([1.0, 0.0, 0.0])
            self.heading = float(np.arctan2(fwd[1], fwd[0]))
        else:
            # робастная подгонка прямой: точки с большой невязкой (ложные детекции, прошедшие триангуляцию
            # по двум камерам) отбрасываются и подгонка повторяется
            A = np.c_[frames - frame, np.ones(len(frames))]
            coef = np.linalg.lstsq(A, points, rcond=None)[0]
            resid = np.linalg.norm(A @ coef - points, axis=1)
            keep = resid <= max(3.0 * np.median(resid), 2.0 * self.s.units_per_meter)
            if keep.sum() < len(frames):
                if keep.sum() < max(3, self.s.init_frames // 2):
                    self._init_buffer = [b for b, k in zip(self._init_buffer, keep) if k]
                    return False
                coef = np.linalg.lstsq(A[keep], points[keep], rcond=None)[0]
            velocity, center = coef[0], coef[1]
            self.heading = float(np.arctan2(velocity[1], velocity[0]))
        self.heading_dir = np.array([np.cos(self.heading), np.sin(self.heading), 0.0])
        self._v_smooth = None
        # Центр маски ≈ середина коробки
        self.state = np.zeros(self.n)
        self.state[:3], self.state[3:6] = center, velocity
        diag = [400.0, 400.0, 200.0, 30.0, 30.0, 5.0]
        if mono:
            diag = [2000.0, 2000.0, 500.0, 60.0, 60.0, 10.0]
        if self.n == 9:
            diag += [3.0, 3.0, 1.0]
        self.P = np.diag(diag) ** 2
        self._confirmed = not mono
        self._init_buffer = []
        return True

    # ---------------------------------------------------------------- измерения

    def _reference_point(self, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Опорная точка (низ передней грани) по центру коробки."""
        forward, _, up = truck_axes(direction[None, :])
        return center + self.dims[0] / 2 * forward[0] - self.dims[2] / 2 * up[0]

    def _offsets(self, direction: np.ndarray) -> np.ndarray:
        """Силуэт (оболочка модели или углы коробки) относительно центра при курсе direction; кеш на тик."""
        key = tuple(np.round(direction, 9))
        if self._offsets_cache[0] != key:
            self._offsets_cache = (key, silhouette_offsets(direction, self.model, self.dims))
        return self._offsets_cache[1]

    def _measure(self, center: np.ndarray, direction: np.ndarray, geometry: ObservationGeometry) -> np.ndarray:
        """Рамка (M, 4) x1, x2, y1, y2 проекции силуэта на каждую камеру наблюдений (точки позади — на z = 1)."""
        points = center + self._offsets(direction)                                            # (K, 3)
        cam = np.einsum("nij,nkj->nki", geometry.rotation, points[None, :, :] - geometry.position[:, None, :])
        z = np.maximum(cam[..., 2], 1.0)
        u = geometry.focal[:, 0:1] * cam[..., 0] / z + geometry.center[:, 0:1]
        v = geometry.focal[:, 1:2] * cam[..., 1] / z + geometry.center[:, 1:2]
        return np.c_[u.min(1), u.max(1), v.min(1), v.max(1)]

    def _apply_limits(self, prior: np.ndarray, posterior: np.ndarray, previous_velocity: np.ndarray,
                      dt: float = 1.0) -> np.ndarray:
        s = self.s
        out = posterior.copy()
        dv = out[3:6] - previous_velocity
        direction = previous_velocity / max(np.linalg.norm(previous_velocity), 1e-9)
        longitudinal = dv @ direction
        lateral = dv - longitudinal * direction
        # неподтверждённый трек (старт по одной камере) может быть далеко от правды: пределы скорости ослаблены;
        # предел сдвига положения растёт с неопределённостью фильтра (уверенный трек — 1.5 м/кадр)
        relax = 1.0 if self._confirmed else 3.0
        std_m = float(np.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0))) / s.units_per_meter if self.P is not None else 0.0
        pos_relax = float(np.clip(std_m / s.limit_relax_std_m, 1.0, 3.0))
        long_max = s.max_longitudinal_mps2 * s.accel_to_units * max(dt, 1e-6) * relax
        lat_max = s.max_lateral_mps2 * s.accel_to_units * max(dt, 1e-6) * relax
        longitudinal = np.clip(longitudinal, -long_max, long_max)
        lat_norm = np.linalg.norm(lateral)
        if lat_norm > lat_max:
            lateral = lateral * lat_max / lat_norm
        out[3:6] = previous_velocity + longitudinal * direction + lateral
        shift = out[:3] - prior[:3]
        norm = np.linalg.norm(shift)
        if norm > s.max_correction_per_frame * pos_relax:
            out[:3] = prior[:3] + shift * s.max_correction_per_frame * pos_relax / norm
        if self.n == 9:
            a_max = max(s.max_longitudinal_mps2, s.max_lateral_mps2) * s.accel_to_units
            a_norm = np.linalg.norm(out[6:9])
            if a_norm > a_max:
                out[6:9] *= a_max / a_norm
        return out

    def _point_measurement(self, prior: np.ndarray, direction: np.ndarray, cameras: dict[int, Camera],
                           detections) -> PointMeasurement | None:
        """3D-точка триангуляции центроидов надёжных детекций (reconstruction/fusion.py)."""
        if self.fusion is None or self.fusion.mode == "none":
            return None
        centroids = {d[0]: d[4] for d in detections if d[5] and d[4] is not None}
        if len(centroids) < 2:
            return None
        return build_point_measurement(cameras, centroids, self.fusion, prior[:3], direction, self.model,
                                       self.dims)

    # ---------------------------------------------------------------- шаг

    def step(self, frame: int, cameras: dict[int, Camera], detections, dt_frames: float | None = None) -> FrameEstimate:
        """
        Один тик. dt_frames — шаг времени в кадрах (по умолчанию — разность номеров кадров,
        минимум 1); при потоковой обработке передавайте (t − t_prev)·fps.
        """
        s = self.s
        for d in detections:
            if self._last_seen.get(d[0], -10 ** 9) < frame - 1:
                self._appeared[d[0]] = frame
            self._last_seen[d[0]] = frame
        if dt_frames is None:
            dt = 1.0 if self._last_frame is None else float(max(frame - self._last_frame, 1))
        else:
            dt = float(max(dt_frames, 1e-3))
        self._last_frame = frame
        nan3 = np.full(3, np.nan)

        if self.state is None:
            if not self._try_initialize(frame, cameras, detections):
                return FrameEstimate(frame, INITIALIZING, nan3, nan3, nan3, np.nan, np.nan, np.nan,
                                     len(detections), 0, 0, -1, dt_frames=dt)
            prior, P_prior = self.state.copy(), self.P.copy()
            previous_velocity = prior[3:6].copy()
        elif (not self._confirmed or self._bad_streak >= s.reinit_after_bad_frames) \
                and self._try_initialize(frame, cameras, detections, allow_mono=False):
            # стерео стало доступно (после старта по одной камере) или трек разошёлся: состояние
            # заменяется триангуляцией — стерео всегда важнее накопленной гипотезы
            prior, P_prior = self.state.copy(), self.P.copy()
            previous_velocity = prior[3:6].copy()
            self._bad_streak = 0
        else:
            previous_velocity = self.state[3:6].copy()
            F = self._transition(dt)
            prior = F @ self.state
            P_prior = F @ self.P @ F.T + self._process_noise(dt)

        state, P = prior, P_prior
        occluded_count, used_cameras = 0, 0
        rejected_cameras: list[int] = []
        point = None
        point_used = False
        nis = float("nan")
        reproj_by_camera: dict[int, float] = {}
        direction = self._direction(prior)
        if detections:
            cam_ids = [d[0] for d in detections]
            geometry = ObservationGeometry.from_cameras([cameras[c] for c in cam_ids])
            observed = np.array([d[1] for d in detections], dtype=float)
            valid = np.array([d[2] for d in detections], dtype=bool)
            predicted = self._measure(prior[:3], direction, geometry)
            # Гейт активен, только пока трек недавно подтверждался ≥2 камерами (одна камера, в том числе
            # ложная, подтверждением не считается — иначе трек «запирается» на ложной гипотезе), и никогда
            # не отбрасывает большинство детекций тика: если с прогнозом не согласно большинство, неправ прогноз
            confident = (self._confirmed and self._last_confirmed is not None
                         and frame - self._last_confirmed <= s.gate_max_age
                         and np.sqrt(P_prior[0, 0] + P_prior[1, 1]) / s.units_per_meter < s.gate_max_std_m)
            rejected = gate_detections(predicted, observed, s) if confident else np.zeros(len(detections), bool)
            if rejected.sum() * 2 > len(detections):
                rejected[:] = False
            rejected_cameras = [int(c) for c, r in zip(cam_ids, rejected) if r]
            occluded = detect_occluded_edges(predicted, observed, valid, s) & ~rejected[:, None]
            usable = valid & ~occluded & np.isfinite(predicted) & ~rejected[:, None]
            if not s.use_edges:
                usable[:] = False
            occluded_count = int(occluded.sum())
            used_cameras = int((usable.sum(axis=1) >= 3).sum())

            accepted = [d for d, r in zip(detections, rejected) if not r]
            point = self._point_measurement(prior, direction, cameras, accepted)
            if point is not None and point.ok:
                r = point.point - prior[:3]
                S = P_prior[:3, :3] + point.covariance
                try:
                    nis = float(r @ np.linalg.solve(S, r))
                except np.linalg.LinAlgError:
                    nis = float("inf")
                point_used = nis <= s.point_gate_nis

            if usable.any() or point_used:
                conf = np.array([max(d[3], 1e-3) for d in detections])
                ramp = np.array([
                    s.reappear_min_weight + (1 - s.reappear_min_weight)
                    * min(1.0, (frame - self._appeared.get(d[0], -10 ** 9)) / max(s.reappear_ramp_frames, 1))
                    for d in detections])
                inflate = np.array([s.edge_inflation_with_point if point_used and c in point.cameras else 1.0
                                    for c in cam_ids])
                mask = usable.ravel()
                observed_flat = observed.ravel()[mask]
                R_edges = np.repeat(s.edge_sigma_px / np.sqrt(conf * ramp) * np.sqrt(inflate), 4)[mask] ** 2
                P_inv = np.linalg.inv(P_prior)
                measure = lambda x: self._measure(x[:3], direction, geometry).ravel()[mask]  # noqa: E731
                n_edges = int(mask.sum())
                H_point = np.zeros((3, self.n))
                H_point[:, :3] = np.eye(3)
                R_point_inv = np.linalg.inv(point.covariance) if point_used else None
                state = prior.copy()
                for _ in range(max(1, s.update_iterations)):
                    information = P_inv.copy()
                    gradient = np.zeros(self.n)
                    if n_edges:
                        Hm = np.zeros((n_edges, self.n))
                        steps = np.zeros(self.n)
                        steps[:3] = 1.0
                        Hm[:, :3] = _numeric_jacobian(measure, state, steps)[:, :3]
                        residual = observed_flat - measure(state)
                        S_diag = np.einsum("ij,jk,ik->i", Hm, P_prior, Hm) + R_edges
                        R = R_edges * np.maximum(1.0, np.abs(residual) / np.sqrt(S_diag) / s.huber_k)
                        information += Hm.T @ (Hm / R[:, None])
                        gradient += Hm.T @ ((residual + Hm @ (state - prior)) / R)
                    if point_used:
                        residual_p = point.point - state[:3]
                        information += H_point.T @ R_point_inv @ H_point
                        gradient += H_point.T @ R_point_inv @ (residual_p + H_point @ (state - prior))
                    new_state = prior + np.linalg.solve(information, gradient)
                    converged = np.max(np.abs(new_state[:3] - state[:3])) < 1.0
                    state = new_state
                    if converged:
                        break
                P = np.linalg.inv(information)
                P = (P + P.T) / 2
                state = self._apply_limits(prior, state, previous_velocity, dt)
            # невязка рёбер после обновления — для планировщика камер и отчёта
            after = self._measure(state[:3], direction, geometry)
            res = np.abs(after - observed)
            for i, c in enumerate(cam_ids):
                m = usable[i]
                reproj_by_camera[int(c)] = float(res[i][m].mean()) if m.any() else float("nan")

        self.state, self.P = state, P
        bad = (point is not None and point.ok and not point_used) or len(rejected_cameras) >= 2
        self._bad_streak = self._bad_streak + 1 if bad else 0
        if used_cameras >= 2 or point_used:
            self._confirmed = True
            self._last_confirmed = frame
        status = MEASURED if (used_cameras or point_used) else (PARTIAL if detections else PREDICTED)
        if status == MEASURED:
            self._last_measured = frame
        since = 0 if self._last_measured is None else frame - self._last_measured

        # ускорение: из состояния (ca) или сглаженная разность скоростей (cv)
        if self.n == 9:
            accel_units = state[6:9]
        else:
            if self._prev_velocity is not None:
                raw = (state[3:6] - self._prev_velocity) / dt
                self._accel_est = 0.7 * self._accel_est + 0.3 * raw
            accel_units = self._accel_est
        self._prev_velocity = state[3:6].copy()

        direction = self._update_heading(state, dt)
        forward, side, up = truck_axes(direction[None, :])
        if self.model is not None:
            points = self.model.to_world(state[:3], forward[0], side[0], up[0],
                                         np.vstack([self.model.csv_point, self.model.cab_center]))
            position, cab_center = points[0], points[1]
        else:
            position = self._reference_point(state[:3], direction)
            cab_center = position - s.cab_center_back * forward[0] + s.cab_center_up * up[0]
        to_mps = s.fps / s.units_per_meter
        speed = float(np.linalg.norm(state[3:6]) * to_mps)
        accel_mps2 = accel_units * s.fps ** 2 / s.units_per_meter
        reproj_vals = [v for v in reproj_by_camera.values() if np.isfinite(v)]
        quality = ("point+edges" if point_used and used_cameras else "point" if point_used
                   else "edges" if used_cameras else "predicted" if detections == [] else "partial")
        if not self._confirmed:
            quality = "mono"                                # трек по одной камере, не подтверждён второй
        return FrameEstimate(
            frame, status, position, state[:3].copy(), cab_center, speed, float(self.heading),
            float(np.sqrt(P[0, 0] + P[1, 1])), len(detections), used_cameras, occluded_count, since,
            rejected_cameras=rejected_cameras, velocity_mps=state[3:6] * to_mps, acceleration_mps2=accel_mps2,
            accel_magnitude_mps2=float(np.linalg.norm(accel_mps2)), covariance=P[:3, :3].copy(),
            point_used=point_used, point_cameras=list(point.cameras) if point_used else [],
            point_nis=nis, point_std_m=point.std_m if point is not None else float("nan"),
            reproj_px=float(np.mean(reproj_vals)) if reproj_vals else float("nan"),
            reproj_by_camera=reproj_by_camera, dt_frames=dt, quality=quality,
        )


def run_tracker(observations: BoxObservations, camera_provider: CameraProvider, frames,
                points: dict | None = None, reliable: dict | None = None,
                settings: TrackerSettings | None = None,
                dimensions: TruckDimensions = DEFAULT_DIMENSIONS, model: TruckModel | None = None,
                fusion: FusionSettings | None = None, camera_filter=None) -> TrackResult:
    """
    Прогон трекера по кадрам по порядку (как в реальном времени).

    points / reliable — {(камера, кадр): (u, v)} и {(камера, кадр): bool} для инициализации
    по триангуляции центров масок и для 3D-измерения (fusion).
    camera_filter(frame, camera) -> bool — какие детекции доступны на кадре (планировщик камер).
    """
    tracker = CabTracker(settings, dimensions, model, fusion)
    by_frame: dict[int, list[int]] = {}
    for i, t in enumerate(observations.times.astype(int)):
        by_frame.setdefault(t, []).append(i)
    result = TrackResult()
    for frame in frames:
        idx = [i for i in by_frame.get(int(frame), [])
               if camera_filter is None or camera_filter(int(frame), int(observations.camera_indices[i]))]
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
