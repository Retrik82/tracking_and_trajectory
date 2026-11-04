"""
Конфигурация потоковой системы: все настраиваемые параметры в одном месте.

Загружается из JSON (`--config file.json`), поверх — переопределения из командной строки
(`--set section.key=value`). Значения по умолчанию — ниже; `StreamConfig.default_json()`
печатает полный JSON, который можно сохранить и править.

Разделы:
  cameras      номера камер, размер кадра, сенсор, FPS, пути к файлам
  streaming    режим (live / replay / detections), очереди, синхронизация, устаревшие кадры
  detector     веса YOLO, imgsz, устройство, пороги, пакетная обработка
  scheduler    адаптивный выбор камер (ACTIVE / PROBE / SLEEP)
  triangulation пороги триангуляции реального времени и согласование точек
  tracker      фильтр Калмана (reconstruction/visibility.py)
  spline       скользящий B-сплайн (reconstruction/streaming_bspline.py)
  output       видео, графики, отчёты
  orientation  курс по изображению: сопоставление силуэта модели с маской (reconstruction/orientation.py)
  road         дорожная плоскость: fixed / online (по уверенным стереокадрам трека) и ограничение основания
  homography   гомография дороги: точка основания -> (x, y) центра (reconstruction/ground.py)
  mono_depth   дальность по известным габаритам модели
  confirmation строгая инициализация (независимые камеры, угол лучей, невязка), передача между камерами
  uncertainty  контроль неопределённости: коррекция после пределов, рост без стерео, систематика

Новые разделы по умолчанию выключены: базовая конфигурация (вариант A) не меняется. Включаются
независимо — для ablation (pipeline/variants.py).
"""

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


def _set_nested(obj, dotted: str, value: str) -> None:
    section, _, key = dotted.partition(".")
    if not key or not hasattr(obj, section):
        raise ValueError(f"Неизвестный параметр {dotted!r}")
    target = getattr(obj, section)
    if not hasattr(target, key):
        raise ValueError(f"Неизвестный параметр {dotted!r}")
    current = getattr(target, key)
    if isinstance(current, bool):
        parsed = value.lower() in ("1", "true", "yes", "да")
    elif isinstance(current, int):
        parsed = int(value)
    elif isinstance(current, float):
        parsed = float(value)
    elif isinstance(current, list):
        parsed = json.loads(value) if value.startswith("[") else [type(current[0])(v) if current else v
                                                                   for v in value.split(",")]
    elif current is None:
        if value.lower() in ("none", "null", ""):
            parsed = None
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
    else:
        parsed = value
    setattr(target, key, parsed)


@dataclass
class CamerasConfig:
    ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    width: int = 1920
    height: int = 1080
    sensor_width_mm: float = 23.76
    sensor_height_mm: float = 13.365
    fps: float = 30.0
    units_per_meter: float = 100.0
    frame_offset: int = 0                      # кадр CSV = кадр видео + смещение
    video_pattern: str = "Way1.FinalImage.CineCameraActor{cam}.mp4"
    pose_pattern: str = "object_transform_CineCameraActor{cam}.csv"
    target_csv: str = "object_transform_SM_AA_track.csv"   # только для оценки
    live_sources: dict = field(default_factory=dict)        # {"1": "rtsp://...", ...} для режима live


@dataclass
class StreamingConfig:
    mode: str = "replay"                       # live | replay (видео в реальном темпе) | detections (готовые детекции)
    realtime_pacing: bool = True               # replay: выдавать кадры с исходной частотой (иначе — как можно быстрее)
    queue_size: int = 2                        # кадров в очереди на камеру; старые вытесняются
    sync_tolerance_ms: float = 20.0            # кадры разных камер считаются одновременными в этом допуске
    max_frame_age_ms: float = 150.0            # кадр старше — устаревший, пропускается
    detector_workers: int = 2                  # потоков детекции (при CPU-инференсе больше не помогает)
    max_frames: int | None = None              # ограничить число тиков (отладка)
    log_every: int = 100                       # кадров между строками лога


@dataclass
class DetectorConfig:
    weights: str = "models/best.pt"
    imgsz: int = 960
    conf: float = 0.25
    device: str | None = None                  # cuda / mps / cpu / None — автоматически
    half: bool = False                         # FP16 (только GPU)
    batch_active: bool = True                  # пакетная обработка активных камер одним вызовом
    min_mask_area_px: float = 400.0
    min_reliable_conf: float = 0.5
    edge_margin_px: float = 2.0


@dataclass
class SchedulerConfig:
    enabled: bool = True
    horizon_frames: int = 15                   # на сколько кадров вперёд прогнозировать видимость
    margin_px: float = 40.0                    # «внутри кадра» с запасом
    sigma_scale: float = 2.0                   # разброс положения для вероятности видимости (в СКО)
    active_threshold: float = 0.6              # вероятность видимости для ACTIVE
    probe_threshold: float = 0.1               # ниже — SLEEP
    hysteresis_frames: int = 6                 # столько кадров подряд условие должно держаться
    probe_period_frames: int = 10              # PROBE: YOLO раз в N кадров (3 FPS при 30)
    sleep_period_frames: int = 90              # SLEEP: редкая контрольная детекция (0 — никогда)
    min_active: int = 2                        # минимум активных камер (если есть кандидаты)
    max_active: int | None = None              # ограничение числа ACTIVE (None — без ограничения)
    occluded_patience_frames: int = 15         # видна геометрически, но не найдена столько кадров → PROBE
    force_std_m: float = 3.0                   # СКО положения больше → активировать все кандидаты
    force_nis: float = 9.0                     # нормированная невязка больше → активировать
    force_reproj_px: float = 60.0              # ошибка репроекции больше → активировать
    border_px: float = 80.0                    # объект у края кадра → добавить камеры
    prefer_geometry: bool = True               # добирать камеры по углу пересечения лучей


@dataclass
class TriangulationConfig:
    mode: str = "model_corrected"              # none | centroid | model_corrected
    sigma_px: float = 8.0                      # СКО точки на изображении (для ковариации 3D-точки)
    max_reproj_px: float = 20.0
    min_ray_angle_deg: float = 2.0             # меньше — глубина не определена, измерение отбрасывается
    max_point_std_m: float = 5.0               # ковариация больше — измерение не используется
    covariance_inflation: float = 1.5          # запас на несовпадение центроида и точки модели
    edge_inflation_with_point: float = 2.0     # рёбра той же камеры коррелированы с центроидом: R_edges × k


@dataclass
class TrackerConfig:
    motion_model: str = "cv"                   # cv (постоянная скорость) | ca (постоянное ускорение)
    accel_noise_mps2: float = 6.0
    lateral_accel_noise_mps2: float = 6.0
    vertical_accel_noise_mps2: float = 1.0
    jerk_noise_mps3: float = 20.0              # для модели ca
    edge_sigma_px: float = 20.0
    huber_k: float = 2.5
    max_longitudinal_mps2: float = 12.0
    max_lateral_mps2: float = 8.0
    max_correction_per_frame_m: float = 1.5
    heading_rate_limit_dps: float = 60.0       # предел скорости поворота курса, °/с
    heading_min_speed_mps: float = 1.0
    init_frames: int = 8
    mono_init: bool = False                    # старт по одной камере (грубо, quality = mono); см. visibility.py
    use_edges: bool = True                     # измерение по рёбрам рамки проекции модели
    occlusion_hold_frames: int = 3             # гистерезис закрытых рёбер: удержание решения, кадров (0 — выкл.)
    occlusion_ramp_frames: int = 8             # затем плавный возврат веса ребра за столько кадров
    truck_fbx: str = "models/CargoTruck.fbx"
    truck_scale: float = 3.186
    box_length_m: float = 16.5                 # коробка, если FBX нет
    box_width_m: float = 2.55
    box_height_m: float = 4.0


@dataclass
class SplineConfig:
    enabled: bool = True
    window_frames: int = 45                    # 1.5 с при 30 FPS
    update_every_frames: int = 4               # ≈8 раз в секунду (IRLS ~1 мс, можно чаще)
    lag_frames: int = 5                        # SMOOTHED TRACK отстаёт на столько кадров
    knot_spacing_frames: float = 6.0
    lambda_acceleration: float = 1e3           # единицы см и кадры; подобрано на синтетике (см. streaming_bspline.py)
    lambda_jerk: float = 1e5
    huber_scale_m: float = 1.0
    max_nfev: int = 25
    deadline_ms: float = 25.0                  # бюджет одной оптимизации; при превышении max_nfev уменьшается
    background: bool = True                    # считать в отдельном потоке


@dataclass
class OutputConfig:
    out_dir: str = "outputs/stream"
    video: bool = False
    video_cameras: list[int] = field(default_factory=list)   # пусто — все
    plots: bool = True
    save_track_csv: bool = True
    performance_report: bool = True


@dataclass
class OrientationConfig:
    enabled: bool = False
    sigma_floor_deg: float = 15.0              # систематика силуэта (модель ≠ маска, склон) в квадратуре к СКО пика
    gate: float = 3.0                          # невязка курса больше gate·√S — отклонить
    max_step_deg: float = 5.0                  # поправка курса за кадр не больше
    process_dps: float = 20.0                  # рост неопределённости курса, °/с
    max_slip_deg: float = 35.0                 # на скорости визуальный курс не дальше этого от направления скорости
    mode: str = "direct"                       # direct — поправка курса; bias — оценка увода кузова от скорости
    bias_decay: float = 0.98                   # bias: затухание увода за кадр без визуального измерения
    slip_speed_mps: float = 3.0
    step_deg: float = 10.0                     # сетка курсов
    min_iou: float = 0.55                      # маска хуже совпадает с моделью — камера не используется
    min_axis_contrast: float = 0.04            # ось не выражена — измерения нет
    min_front_margin: float = 0.03             # «перед/зад» не различим — только ось (±180°)


@dataclass
class RoadConfig:
    mode: str = "none"                         # none | fixed | online
    height_m: float = 0.0                      # fixed: высота дороги
    sigma_m: float = 0.3                       # СКО высоты дороги (нижняя граница)
    radius_m: float = 30.0                     # online: радиус локальной плоскости
    min_points: int = 15
    constraint: bool = False                   # слабое измерение «основание на дороге»
    only_without_stereo: bool = True


@dataclass
class HomographyConfig:
    enabled: bool = False
    sigma_px: float = 6.0                      # СКО точки основания на изображении
    inflation: float = 2.0                     # корреляция с рёбрами той же камеры
    max_range_m: float = 150.0                 # рабочая область от камеры
    only_without_stereo: bool = True


@dataclass
class MonoDepthConfig:
    enabled: bool = False
    rel_sigma: float = 0.08                    # относительная ошибка дальности (модель ≠ силуэт)
    only_without_stereo: bool = True
    shape_tol: float = 0.15                    # ширина и высота рамки дают разную дальность — не использовать


@dataclass
class ConfirmationConfig:
    init_min_cameras: int = 2                  # независимых камер для инициализации
    init_min_ray_angle_deg: float = 0.0        # 0 — без проверки угла
    init_max_reproj_px: float = 0.0            # 0 — без проверки невязки
    handoff_gate_chi2: float = 0.0             # 0 — без проверки камеры, только что увидевшей грузовик
    aux_gate_chi2: float = 9.21                # гейт вспомогательных измерений (дорога, гомография, дальность)


@dataclass
class UncertaintyConfig:
    clip_covariance: bool = False              # P += δδᵀ после физических пределов
    no_stereo_q_growth_per_s: float = 0.0      # процессный шум растёт без стерео
    no_stereo_q_max: float = 10.0
    depth_sigma_growth_mps: float = 0.0        # рост СКО вдоль луча единственной камеры, м/с
    model_sigma_px: float = 0.0                # несовпадение модели с силуэтом, px
    calibration_sigma_px: float = 0.0          # погрешность калибровки/позы камеры, px
    heading_in_edges: bool = False             # неопределённость курса — в ковариацию рёбер
    heading_sigma_floor_deg: float = 0.0       # нижняя граница СКО курса (курс по скорости занижает её)
    heading_turn_lag_s: float = 0.0            # СКО курса += скорость поворота × запаздывание
    model_length_sigma_m: float = 0.0          # погрешность длины модели -> в ковариацию точки вывода (вдоль оси)
    lost_after_frames: int = 90                # кадров только прогноза — уровень «потерян»


@dataclass
class StreamConfig:
    cameras: CamerasConfig = field(default_factory=CamerasConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    spline: SplineConfig = field(default_factory=SplineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    road: RoadConfig = field(default_factory=RoadConfig)
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    mono_depth: MonoDepthConfig = field(default_factory=MonoDepthConfig)
    confirmation: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

    @classmethod
    def load(cls, path: Path | str | None = None, overrides: list[str] | None = None) -> "StreamConfig":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for section_field in fields(cls):
                section = data.get(section_field.name)
                if section is None:
                    continue
                target = getattr(cfg, section_field.name)
                for key, value in section.items():
                    if not hasattr(target, key):
                        raise ValueError(f"Неизвестный параметр {section_field.name}.{key}")
                    setattr(target, key, value)
        for item in overrides or []:
            key, _, value = item.partition("=")
            _set_nested(cfg, key.strip(), value.strip())
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def default_json(cls) -> str:
        return json.dumps(cls().to_dict(), ensure_ascii=False, indent=2)

    def copy(self) -> "StreamConfig":
        return StreamConfig.load(None, [])._update_from(self.to_dict())

    def _update_from(self, data: dict) -> "StreamConfig":
        for section_field in fields(self):
            target = getattr(self, section_field.name)
            for key, value in data.get(section_field.name, {}).items():
                setattr(target, key, json.loads(json.dumps(value)))
        return self


def describe(cfg: StreamConfig) -> str:
    """Короткая строка для лога."""
    s = cfg
    return (f"режим {s.streaming.mode}, камеры {s.cameras.ids}, планировщик {'вкл' if s.scheduler.enabled else 'выкл'}, "
            f"триангуляция {s.triangulation.mode}, модель движения {s.tracker.motion_model}, "
            f"сплайн {'вкл' if s.spline.enabled else 'выкл'} (окно {s.spline.window_frames}, задержка {s.spline.lag_frames})")


assert is_dataclass(StreamConfig)
