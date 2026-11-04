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
   ребро, ушедшее внутрь при совпадающем противоположном, исключается. Решение удерживается
   несколько кадров и снимается плавно (иначе дребезг и зигзаг оценки).
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
from reconstruction.fusion import (
    FusionSettings,
    PointMeasurement,
    build_point_measurement,
    ray_angles_deg,
    silhouette_offsets,
)
from reconstruction.ground import OnlineRoadModel, RoadPlane, ground_homography, ground_jacobian, pixel_to_ground
from reconstruction.objective import CameraProvider, ObservationGeometry
from reconstruction.orientation import OrientationSettings, SilhouetteMatcher, estimate_heading
from reconstruction.silhouette_fit import SilhouetteFitSettings, fit_pose, select_views
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
    # Гистерезис закрытых рёбер. Решение «ребро закрыто» на соседних кадрах дребезжало
    # (3-2-3-2 закрытых ребра на кадрах 417–429, 104 смены за видео), набор измерений менялся
    # каждый кадр, и оценка шла зигзагом. Ребро, признанное закрытым, держится закрытым
    # occlusion_hold_frames кадров после последнего срабатывания, затем возвращается с весом,
    # растущим occlusion_ramp_frames кадров. На YOLO: точка CSV 2.20/7.14 → 1.95/6.19 м,
    # ускорение по траектории p95 91 → 51 м/с². 0 — выключено.
    occlusion_hold_frames: int = 3
    occlusion_ramp_frames: int = 8
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

    # ------------------------------------------------------------------ доработки (все выключены по умолчанию:
    # базовый вариант A не меняется; включаются независимо для ablation, см. pipeline/config.py)
    # Курс по изображению (reconstruction/orientation.py): сопоставление силуэта модели с маской
    visual_heading: bool = False
    visual_heading_sigma_floor_deg: float = 15.0   # систематика: модель ≠ силуэт, склон, тень в маске
    visual_heading_gate: float = 3.0               # |невязка| > gate·√S — измерение отклоняется
    visual_heading_max_step_deg: float = 5.0       # поправка курса за кадр не больше (одна маска не разворачивает)
    heading_process_dps: float = 20.0              # случайное блуждание курса (рост его дисперсии), °/с
    # Физическое ограничение: на скорости кузов не стоит поперёк движения (увод < max_slip). Визуальный курс,
    # отклонённый от направления скорости больше, отклоняется — иначе ошибка «перед/зад» разворачивает трек
    visual_heading_max_slip_deg: float = 35.0
    # direct — поправка курса напрямую (курс по скорости тянет обратно каждый кадр: зигзаг точки вывода);
    # bias — курс по силуэту оценивает отклонение кузова от направления скорости (увод), цель курса по скорости
    # = направление скорости + увод; без визуальных измерений увод затухает
    visual_heading_mode: str = "direct"
    visual_bias_decay: float = 0.98
    visual_heading_slip_speed_mps: float = 3.0
    # Дорожная плоскость: none | fixed (road_height) | online (по уверенным стереокадрам самого трека)
    road_mode: str = "none"
    road_height: float = 0.0                       # ед. мира, для fixed
    road_sigma: float = 30.0                       # СКО высоты дороги (нижняя граница), ед. мира
    road_radius: float = 3000.0                    # online: радиус локальной плоскости, ед. мира
    road_min_points: int = 15
    road_constraint: bool = False                  # слабое измерение «основание на дороге» (z центра)
    road_only_without_stereo: bool = True          # …только когда нет стерео
    # Гомография дороги: точка основания (низ рамки + поправка ракурса по модели) -> (x, y) центра
    homography: bool = False
    homography_sigma_px: float = 6.0
    homography_inflation: float = 2.0              # коррелирована с рёбрами той же камеры
    homography_max_range: float = 15000.0          # рабочая область, ед. мира от камеры
    homography_only_without_stereo: bool = True
    # Монокулярная дальность по габаритам модели (размер рамки против проекции модели)
    mono_depth: bool = False
    mono_depth_rel_sigma: float = 0.08             # относительная ошибка (модель ≠ силуэт, маска)
    mono_depth_only_without_stereo: bool = True
    aux_gate_chi2: float = 9.21                    # гейт вспомогательных измерений (χ², 99 % для 2 ст. свободы)
    # Строгая инициализация: независимые камеры, угол лучей, невязка (0 / inf — как раньше)
    init_min_cameras: int = 2
    init_min_ray_angle_deg: float = 0.0
    init_max_reproj_px: float = float("inf")
    # Передача между камерами: детекция камеры, только что увидевшей грузовик, сверяется с ожидаемой
    # проекцией трека по χ² центра рамки (0 — выключено)
    handoff_gate_chi2: float = 0.0
    # Неопределённость
    clip_covariance: bool = False                  # после физических пределов P += δ δᵀ (δ — внесённая поправка)
    no_stereo_q_growth_per_s: float = 0.0          # Q × (1 + g · секунд без стерео), не больше no_stereo_q_max
    no_stereo_q_max: float = 10.0
    depth_sigma_growth_mps: float = 0.0            # без стерео: рост СКО вдоль луча камеры, м за секунду
    model_sigma_px: float = 0.0                    # несовпадение модели с силуэтом (в квадратуре к СКО ребра)
    calibration_sigma_px: float = 0.0              # погрешность калибровки и позы камеры, px
    heading_in_edges: bool = False                 # неопределённость курса -> в ковариацию рёбер
    # Неопределённость курса по скорости занижена: курс отстаёт от поворота (сглаживание, предел скорости
    # поворота) и шумит при малой скорости. Нижняя граница и член «запаздывание × скорость поворота»
    heading_sigma_floor_deg: float = 0.0
    heading_turn_lag_s: float = 0.0
    model_length_sigma: float = 0.0                # погрешность длины модели (ед. мира) -> вдоль оси в ковариацию точки вывода
    mono_depth_shape_tol: float = 0.15             # |w-ratio − h-ratio| / ratio больше — рамка искажена, дальности нет
    lost_after_frames: int = 90                    # столько кадров только прогноза — уровень «потерян»
    # Подгонка модели к контуру масок ≥2 камер (reconstruction/silhouette_fit.py): измерение (x, y) и курса
    # Срезанная рамка по размеру: рамка камеры уже (ниже) проекции модели больше чем в size_ratio раз — край
    # срезан перекрытием; срезанным считается ребро, сильнее отклонившееся от прогноза (отставание прогноза сдвигает
    # оба ребра одинаково, перекрытие — одно). 0 — выключено
    occlusion_size_ratio: float = 0.0
    # Физический предел поворота для ИТОГОВОГО курса (включая поправки по силуэту и контуру, которые иначе
    # обходят предел): за кадр не больше (v / R_min + heading_rate_floor) · dt. Без него курс на медленном повороте
    # дёргался до 11°/кадр, а точка вывода в 11 м от центра прыгала на ~2 м. 0 — выключено
    heading_output_rate_limit: bool = False
    heading_rate_floor_dps: float = 5.0
    heading_accel_limit_dps2: float = 0.0          # предел изменения скорости поворота (руль не перекладывается
                                                   # мгновенно туда-обратно — убирает зигзаг курса), °/с²; 0 — выкл.
    silhouette_fit: bool = False
    # measure — (x, y) и курс в фильтр; heading — только курс; output — фильтр как без подгонки, а точка вывода
    # на кадре берётся из подгонки к контуру (начальное приближение — оценка фильтра), если она удалась
    silhouette_mode: str = "measure"
    silhouette_gate_chi2: float = 13.8             # гейт (x, y) по NIS (χ², 2 ст. свободы)
    silhouette_heading_gate: float = 3.0           # гейт курса: |невязка| ≤ k·√S

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


def monocular_range(predicted_range: float, predicted_box: np.ndarray, observed_box: np.ndarray,
                    shape_tol: float = 0.15, min_px: float = 8.0) -> float:
    """
    Дальность центра по известным габаритам: рамка проекции модели при прогнозной дальности r и курсе
    против рамки детектора — размер обратно пропорционален дальности, r_изм = r · (w_пр + h_пр) / (w + h).
    Курс и перспектива учтены тем, что прогнозная рамка — проекция самой 3D-модели, а не формула «H / h».
    NaN, если рамка мала или ширина и высота дают разную дальность больше shape_tol (рамка обрезана,
    закрыта или курс неверен — такое измерение исказило бы глубину). Рамки x1 x2 y1 y2.
    """
    ow, oh = observed_box[1] - observed_box[0], observed_box[3] - observed_box[2]
    pw, ph = predicted_box[1] - predicted_box[0], predicted_box[3] - predicted_box[2]
    if min(ow, oh) <= min_px or min(pw, ph) <= 0:
        return float("nan")
    if abs(pw / ow - ph / oh) > shape_tol * (pw + ph) / (ow + oh):
        return float("nan")
    return float(predicted_range * (pw + ph) / (ow + oh))


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
    # --- уровень наблюдения и неопределённость (доработка)
    observation_level: str = "none"       # multi | multi_partial | single | road | predicted | lost | initializing
    reference_covariance: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan))  # точки вывода, ед.²
    heading_std: float = float("nan")     # СКО курса, радианы
    visual_heading: float = float("nan")  # измеренный по силуэту курс, радианы (NaN — нет измерения)
    visual_heading_used: bool = False
    aux_used: list = field(default_factory=list)       # вспомогательные измерения: road, homography:N, depth:N
    aux_rejected: int = 0
    aux_nis: list = field(default_factory=list)
    frames_since_stereo: int = -1
    clip_correction: float = 0.0          # величина поправки физических пределов, ед. мира

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
        self._occluded_at: dict[tuple[int, int], int] = {}     # (камера, ребро) -> последнее «закрыто»
        self._last_confirmed: int | None = None             # последний кадр с подтверждением ≥2 камерами
        self._bad_streak = 0                                # кадров подряд, когда стерео не согласуется с треком
        self._last_frame: int | None = None
        self._prev_velocity: np.ndarray | None = None
        self._accel_est = np.zeros(3)                       # сглаженное ускорение (модель cv), ед./кадр²
        self._offsets_cache: tuple = (None, None)
        # --- доработка: курс с дисперсией, дорога, учёт стерео, статистика инициализаций
        self.heading_var = np.radians(30.0) ** 2
        self._matcher = SilhouetteMatcher(OrientationSettings(), model, None if model is not None else self.dims)
        self.orientation_settings = self._matcher.s
        prior = RoadPlane.horizontal(self.s.road_height, self.s.road_sigma) if self.s.road_mode == "fixed" else None
        self.road = OnlineRoadModel(self.s.road_radius, self.s.road_min_points, sigma_floor=self.s.road_sigma,
                                    prior=prior) \
            if self.s.road_mode in ("fixed", "online") or self.s.road_constraint or self.s.homography else None
        self._last_stereo: int | None = None
        self.initializations = 0                            # сколько раз трек (пере)инициализирован
        self.init_rejected = 0                              # стереоточек отклонено строгой проверкой
        self._prev_heading: float | None = None
        self._visual_bias = 0.0                             # увод кузова от направления скорости по силуэту, рад
        self._bias_updated = False
        self._sf = SilhouetteFitSettings()
        self._out_heading: float | None = None              # курс прошлого кадра (для предела поворота)
        self._out_yaw_rate = 0.0                            # скорость поворота итогового курса, рад/кадр
        self._verified: set[int] = set()                    # камеры, чья детекция после появления согласовалась с треком
        self._yaw_rate = 0.0                                # рад/с, сглаженная
        self._heading_var_report = self.heading_var         # дисперсия курса с систематикой (для весов и вывода)

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
            if self.s.visual_heading and self.s.visual_heading_mode == "bias":
                target = float(_wrap(target + self._visual_bias))
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
        self._update_heading_variance(state, dt, speed)
        if not self._bias_updated:
            self._visual_bias *= self.s.visual_bias_decay ** max(dt, 0.0)
        self._bias_updated = False
        return self.heading_dir

    def _update_heading_variance(self, state: np.ndarray, dt: float, speed: float) -> None:
        """
        Дисперсия курса: растёт как случайное блуждание (heading_process_dps), уменьшается информацией
        направления скорости — дисперсия направления = поперечная дисперсия скорости / скорость²
        (при малой скорости направление скорости неинформативно). Не меньше (0.5°)², не больше (90°)².
        """
        s = self.s
        self.heading_var += (np.radians(s.heading_process_dps) / s.fps * max(dt, 1e-6)) ** 2
        v = state[3:5]
        norm = float(np.linalg.norm(v))
        if self.P is not None and speed >= s.min_heading_speed_mps and norm > 1e-9:
            lateral = np.array([-v[1], v[0]]) / norm
            var_dir = float(lateral @ self.P[3:5, 3:5] @ lateral) / norm ** 2
            if var_dir > 0:
                self.heading_var = 1.0 / (1.0 / self.heading_var + 1.0 / var_dir)
        # скорость поворота курса (сглаженная) — для члена запаздывания
        if self._prev_heading is not None:
            step = float(_wrap(self.heading - self._prev_heading)) / max(dt, 1e-6)
            self._yaw_rate = 0.8 * self._yaw_rate + 0.2 * abs(step) * s.fps
        self._prev_heading = self.heading
        floor = np.radians(s.heading_sigma_floor_deg) ** 2 + (self._yaw_rate * s.heading_turn_lag_s) ** 2
        self._heading_var_report = float(np.clip(self.heading_var + floor, np.radians(0.5) ** 2, np.radians(90.0) ** 2))
        self.heading_var = float(np.clip(self.heading_var, np.radians(0.5) ** 2, np.radians(90.0) ** 2))

    def _visual_heading_update(self, frame: int, cameras: dict[int, Camera], detections, center: np.ndarray,
                               clean: np.ndarray):
        """
        Курс по силуэту (orientation.py) по камерам с чистыми рамками (все рёбра в кадре и не закрыты, не
        отброшены гейтом) — объединение с курсом трека по весам дисперсий. Если «перед/зад» не определён,
        из двух направлений берётся ближайшее к курсу трека. Невязка больше gate·√S — измерение отклоняется
        (одна ошибочная маска не разворачивает модель); поправка за кадр ограничена.
        """
        s = self.s
        views = [(d[0], cameras[d[0]], d[6], max(float(d[3]), 1e-3)) for d, ok in zip(detections, clean)
                 if ok and len(d) > 6 and d[6] is not None]
        if not views:
            return None, False
        self._matcher.slope = float(self.heading_dir[2] / max(np.hypot(*self.heading_dir[:2]), 1e-9))
        m = estimate_heading(self._matcher, views, center)
        if not m.ok:
            return m, False
        innovation = float(_wrap(m.heading - self.heading))
        if m.axis_only and abs(innovation) > np.pi / 2:
            innovation = float(_wrap(innovation + np.pi))
        v = self.state[3:5] if self.state is not None else np.zeros(2)
        if self._speed_mps(self.state) >= s.visual_heading_slip_speed_mps:
            new = self.heading + innovation
            slip = abs(float(_wrap(new - np.arctan2(v[1], v[0]))))
            if slip > np.radians(s.visual_heading_max_slip_deg):
                return m, False
        var_m = m.sigma ** 2 + np.radians(s.visual_heading_sigma_floor_deg) ** 2
        S = self._heading_var_report + var_m
        if abs(innovation) > s.visual_heading_gate * np.sqrt(S):
            return m, False
        gain = self._heading_var_report / S
        step = float(np.clip(gain * innovation, -np.radians(s.visual_heading_max_step_deg),
                             np.radians(s.visual_heading_max_step_deg)))
        self.heading = float(_wrap(self.heading + step))
        if s.visual_heading_mode == "bias":
            lim = np.radians(s.visual_heading_max_slip_deg)
            self._visual_bias = float(np.clip(self._visual_bias + step, -lim, lim))
            self._bias_updated = True
        self.heading_var = float((1 - gain) * self.heading_var)
        self._heading_var_report = float((1 - gain) * self._heading_var_report)
        slope = self.heading_dir[2] / max(np.hypot(*self.heading_dir[:2]), 1e-9)
        d = np.array([np.cos(self.heading), np.sin(self.heading), slope])
        self.heading_dir = d / np.linalg.norm(d)
        return m, True

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
            if res.ok and self._confirm_point(res, [cameras[d[0]] for d in reliable], [d[0] for d in reliable]):
                self._init_buffer.append((frame, res.point))
        elif len(reliable) == 1 and self.s.mono_init and allow_mono:
            p = self._mono_point(cameras[reliable[0][0]], reliable[0])
            if p is not None:
                self._mono_buffer.append((frame, p))
        # Разреженные тики (бюджет YOLO на слабом CPU): стерео-точки приходят раз в несколько
        # кадров, и окно в 3·init_frames КАДРОВ не вмещает init_frames точек по построению.
        # Окно и требуемое число точек адаптируются к фактическому шагу измерений; при полном
        # темпе (шаг ≤ 2 кадра) поведение в точности прежнее. Линейность на окне ~2 с терпима:
        # инициализация лишь сеет фильтр (СКО позиции 4 м), дальше правят обычные обновления.
        def cadence(buf):
            gaps = np.diff([f for f, _ in buf]) if len(buf) >= 2 else []
            return float(np.median(gaps)) if len(gaps) else 1.0

        def sparse_window(needed_pts: float, step: float) -> int:
            # окно должно вмещать needed точек при фактическом шаге с запасом на промахи
            # детектора (×1.6), но не длиннее 4 с — дальше линейная модель буфера не годится
            return int(np.ceil(min(4.0 * self.s.fps, 1.6 * needed_pts * step)))

        spacing = cadence(self._init_buffer)
        if spacing <= 2.0:
            window, needed = 3 * self.s.init_frames, self.s.init_frames
        else:
            needed = max(4, self.s.init_frames // 2)
            window = sparse_window(needed, spacing)
        # моно-буфер: тот же принцип — при полном темпе прежние 2·init_frames точек,
        # при редких тиках порог и окно, достижимые при фактическом шаге измерений
        mono_spacing = cadence(self._mono_buffer)
        if mono_spacing <= 2.0:
            mono_window, mono_needed = window, 2 * self.s.init_frames
        else:
            mono_needed = max(4, self.s.init_frames // 2)
            mono_window = sparse_window(mono_needed, mono_spacing)
        self._init_buffer = [(f, p) for f, p in self._init_buffer if frame - f < window]
        self._mono_buffer = [(f, p) for f, p in self._mono_buffer if frame - f < mono_window]
        mono = False
        buffer = self._init_buffer
        if len(buffer) < needed:
            if not allow_mono or self._init_buffer or len(self._mono_buffer) < mono_needed:
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
                if keep.sum() < max(3, needed // 2):
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
        self.initializations += 1
        self.heading_var = np.radians(30.0 if not mono else 90.0) ** 2
        self._heading_var_report = self.heading_var
        return True

    def _confirm_point(self, res, cams: list[Camera], cam_ids: list[int]) -> bool:
        """
        Строгая проверка стереоточки для инициализации: независимых (разных) камер-инлаеров не меньше
        init_min_cameras, лучи пересекаются под углом не меньше init_min_ray_angle_deg (почти параллельные
        лучи глубины не дают), невязка инлаеров не больше init_max_reproj_px, точка перед камерами.
        """
        s = self.s
        inl = [i for i, k in enumerate(res.inliers) if k]
        ids = {cam_ids[i] for i in inl}
        ok = len(ids) >= max(s.init_min_cameras, 2)
        if ok and s.init_min_ray_angle_deg > 0:
            angles = ray_angles_deg([cams[i] for i in inl], res.point)
            ok = len(angles) > 0 and float(angles.max()) >= s.init_min_ray_angle_deg
        if ok and np.isfinite(s.init_max_reproj_px):
            err = np.asarray(res.reprojection_errors, float)[inl]
            ok = bool(np.all(np.isfinite(err)) and err.max() <= s.init_max_reproj_px)
        strict = s.init_min_cameras > 2 or s.init_min_ray_angle_deg > 0 or np.isfinite(s.init_max_reproj_px)
        if ok and strict:
            ok = all(cams[i].depth(res.point) > 0 for i in inl)
        if not ok:
            self.init_rejected += 1
        return ok

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
        # предел — «на кадр» (имя параметра): при разреженных тиках (dt кадров за шаг) ёмкость
        # коррекции масштабируется на dt, иначе фильтр не успевает за накопленной ошибкой прогноза
        shift = out[:3] - prior[:3]
        norm = np.linalg.norm(shift)
        max_shift = s.max_correction_per_frame * pos_relax * max(dt, 1.0)
        if norm > max_shift:
            out[:3] = prior[:3] + shift * max_shift / norm
        if self.n == 9:
            a_max = max(s.max_longitudinal_mps2, s.max_lateral_mps2) * s.accel_to_units
            a_norm = np.linalg.norm(out[6:9])
            if a_norm > a_max:
                out[6:9] *= a_max / a_norm
        return out

    def _occlusion_hysteresis(self, frame: int, detections, occluded: np.ndarray) -> np.ndarray:
        """Вес рёбер (M, 4): 0 — закрыто (с удержанием решения), затем плавный возврат к 1."""
        s = self.s
        weight = (~occluded).astype(float)
        if s.occlusion_hold_frames <= 0:
            return weight
        for i, d in enumerate(detections):
            for k in range(4):
                key = (d[0], k)
                if occluded[i, k]:
                    self._occluded_at[key] = frame
                last = self._occluded_at.get(key)
                if last is None:
                    continue
                age = frame - last - s.occlusion_hold_frames
                if age >= s.occlusion_ramp_frames:
                    del self._occluded_at[key]
                else:
                    weight[i, k] = max(0.0, age / s.occlusion_ramp_frames)
        return weight

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
    # ---------------------------------------------------------------- доработка: вспомогательные измерения

    def _size_cut_edges(self, predicted: np.ndarray, observed: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Рёбра (M, 4), срезанные перекрытием, по отношению размеров рамки детектора и прогнозной рамки."""
        out = np.zeros_like(valid)
        r = self.s.occlusion_size_ratio
        for i in range(len(observed)):
            for a, b in ((0, 1), (2, 3)):
                pw, ow = predicted[i, b] - predicted[i, a], observed[i, b] - observed[i, a]
                if pw <= 0 or ow >= r * pw:
                    continue
                da, db = observed[i, a] - predicted[i, a], predicted[i, b] - observed[i, b]   # внутрь — положительно
                k = a if da > db else b
                if valid[i, k]:
                    out[i, k] = True
        return out

    def _silhouette_update(self, cameras, detections, prior, P_prior, valid, occluded, rejected):
        """
        Подгонка модели к контуру чистых масок (≥2 камер). Курс исправляется сразу (скалярный фильтр, гейт),
        (x, y) возвращается как вспомогательное измерение с гейтом NIS. Возврат: (имя, функция, R, курс_обновлён)
        или (None, None, None, курс_обновлён); None — подгонки не было.
        """
        s = self.s
        clean = valid.all(axis=1) & ~occluded.any(axis=1) & ~rejected
        masks = {d[0]: d[6] for d, ok in zip(detections, clean) if ok and len(d) > 6 and d[6] is not None}
        if len(masks) < self._sf.min_views:
            return None
        slope = float(self.heading_dir[2] / max(np.hypot(*self.heading_dir[:2]), 1e-9))
        views = select_views(cameras, masks, prior[:3], self.heading, slope, self.model, self.dims, self._sf)
        fit = fit_pose(views, prior[:3], self.heading, slope, self.model, self.dims, self._sf)
        if not fit.ok:
            return None
        heading_updated = False
        innovation = float(_wrap(fit.heading - self.heading))
        S_h = self._heading_var_report + fit.covariance[2, 2]
        if abs(innovation) <= s.silhouette_heading_gate * np.sqrt(S_h):
            gain = self._heading_var_report / S_h
            self.heading = float(_wrap(self.heading + gain * innovation))
            self.heading_var = float((1 - gain) * self.heading_var)
            self._heading_var_report = float((1 - gain) * self._heading_var_report)
            d = np.array([np.cos(self.heading), np.sin(self.heading), slope])
            self.heading_dir = d / np.linalg.norm(d)
            heading_updated = True
        Hm = np.zeros((2, self.n))
        Hm[:, :2] = np.eye(2)
        y = np.array([fit.x, fit.y])
        R = fit.covariance[:2, :2]
        r0 = y - prior[:2]
        try:
            nis = float(r0 @ np.linalg.solve(P_prior[:2, :2] + R, r0))
        except np.linalg.LinAlgError:
            nis = float("inf")
        if nis > s.silhouette_gate_chi2:
            return (None, None, None, heading_updated)
        return ("silhouette", (lambda x, Hm=Hm, y=y: (y - Hm @ x, Hm)), R, heading_updated)

    def _plane_at(self, xy: np.ndarray) -> RoadPlane | None:
        return None if self.road is None else self.road.local_plane(np.asarray(xy, float)[:2])

    def expected_view(self, camera: Camera, P: np.ndarray | None = None) -> dict:
        """
        Ожидаемая проекция текущего состояния в камеру (передача между камерами): рамка x1 x2 y1 y2,
        центр рамки и его ковариация (J P Jᵀ, px²), видна ли рамка в кадре, ожидаемый курс.
        """
        if self.state is None:
            return {"visible": False}
        P = self.P if P is None else P
        geometry = ObservationGeometry.from_cameras([camera])
        direction = self.heading_dir
        box = self._measure(self.state[:3], direction, geometry)[0]
        center_px = np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2])
        f = lambda x: (lambda b: np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2]))(  # noqa: E731
            self._measure(x[:3], direction, geometry)[0])
        J = _numeric_jacobian(f, self.state[:3].copy(), np.full(3, 1.0))
        cov = J @ P[:3, :3] @ J.T
        s = self.s
        inside = bool(box[1] > 0 and box[0] < s.image_width and box[3] > 0 and box[2] < s.image_height
                      and camera.depth(self.state[:3]) > 0)
        return {"visible": inside, "box": box, "center_px": center_px, "center_cov": cov,
                "heading": float(self.heading), "heading_std": float(np.sqrt(self._heading_var_report))}

    def _handoff_rejects(self, frame: int, cameras: dict[int, Camera], detections, P_prior: np.ndarray,
                         predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """
        Камера, (снова) увидевшая грузовик, проверяется по ожидаемой проекции трека, пока её детекция не
        согласуется с ней: χ² центра рамки с ковариацией J P Jᵀ + (СКО ребра² + (0.15·размер)²) больше
        handoff_gate_chi2 — отклонить (не больше половины детекций тика — см. step).
        Новый трек из-за перехода между камерами не создаётся: состояние в мировой системе общее.
        """
        s = self.s
        out = np.zeros(len(detections), bool)
        if s.handoff_gate_chi2 <= 0 or not self._confirmed:
            return out
        for i, d in enumerate(detections):
            # камера проверяется, пока не дала согласованной с прогнозом детекции после появления
            if self._appeared.get(d[0]) == frame:
                self._verified.discard(d[0])
            if d[0] in self._verified:
                continue
            view = self.expected_view(cameras[d[0]], P_prior)
            if not view.get("visible", False):
                continue
            size = max(predicted[i, 1] - predicted[i, 0], predicted[i, 3] - predicted[i, 2], 1.0)
            S = view["center_cov"] + np.eye(2) * (s.edge_sigma_px ** 2 + (0.15 * size) ** 2)
            r = np.array([(observed[i, 0] + observed[i, 1]) / 2, (observed[i, 2] + observed[i, 3]) / 2]) \
                - np.array([(predicted[i, 0] + predicted[i, 1]) / 2, (predicted[i, 2] + predicted[i, 3]) / 2])
            out[i] = float(r @ np.linalg.solve(S, r)) > s.handoff_gate_chi2
            if not out[i]:
                self._verified.add(d[0])
        return out

    def _pixel_sigma2(self, conf: float) -> float:
        s = self.s
        return s.homography_sigma_px ** 2 / max(conf, 1e-3) + s.model_sigma_px ** 2 + s.calibration_sigma_px ** 2

    def _aux_measurements(self, prior: np.ndarray, P_prior: np.ndarray, direction: np.ndarray,
                          cameras: dict[int, Camera], detections, predicted: np.ndarray, observed: np.ndarray,
                          usable: np.ndarray, stereo: bool) -> tuple[list, int, list]:
        """
        Вспомогательные измерения (каждое: имя, функция x -> (невязка, H), R):
        - road: основание на дорожной плоскости, z центра = z дороги + H/2 (1 ст. свободы);
        - homography:N: точка основания по гомографии -> (x, y) центра (2 ст. свободы);
        - depth:N: дальность центра от камеры по размеру рамки против проекции модели (1 ст. свободы).
        Каждое проходит гейт по NIS против прогноза; конфликт гомографии и дальности одной камеры —
        остаётся то, что лучше согласуется с прогнозом (не слепое усреднение).
        """
        s = self.s
        out, nis_list = [], []
        n = self.n
        plane = self._plane_at(prior[:2]) if (s.road_constraint or s.homography) else None
        half_h = float(self.dims[2]) / 2
        heading_std = float(np.sqrt(self._heading_var_report))
        if s.road_constraint and plane is not None and (not s.road_only_without_stereo or not stereo):
            sx, sy = plane.slope()
            Hm = np.zeros((1, n))
            Hm[0, :3] = [-sx, -sy, 1.0]
            y = np.array([plane.point[2] - sx * plane.point[0] - sy * plane.point[1] + half_h])
            R = np.array([[max(plane.sigma, s.road_sigma) ** 2]])
            out.append(("road", (lambda x, Hm=Hm, y=y: (y - Hm @ x, Hm)), R))
        by_cam: dict[int, dict] = {}
        for i, d in enumerate(detections):
            cam = cameras[d[0]]
            conf = max(float(d[3]), 1e-3)
            if s.homography and plane is not None and (not s.homography_only_without_stereo or not stereo) \
                    and usable[i, 0] and usable[i, 1] and usable[i, 3]:
                h = ground_homography(cam, plane)
                foot = prior[:3] - half_h * plane.normal
                pF = cam.project(foot)
                pB = np.array([(predicted[i, 0] + predicted[i, 1]) / 2, predicted[i, 3]])
                q = np.array([(observed[i, 0] + observed[i, 1]) / 2, observed[i, 3]]) + (pF - pB)
                X, ok = pixel_to_ground(h, q[None, :], s.homography_max_range)
                if h.ok and ok[0] and np.all(np.isfinite(pF)):
                    J = ground_jacobian(h, q)[:2]
                    lever = float(np.linalg.norm(pF - pB)) * heading_std
                    sig2 = (self._pixel_sigma2(conf) + lever ** 2) * s.homography_inflation
                    ray = X[0] - cam.position
                    g = ray[:2] / max(abs(ray[2]), 1e-6)
                    R = J @ J.T * sig2 + np.outer(g, g) * max(plane.sigma, s.road_sigma) ** 2
                    Hm = np.zeros((2, n))
                    Hm[:, :2] = np.eye(2)
                    y = X[0, :2] + 0.0
                    by_cam.setdefault(d[0], {})["homography"] = (f"homography:{d[0]}",
                                                                 (lambda x, Hm=Hm, y=y: (y - Hm @ x, Hm)), R)
            if s.mono_depth and (not s.mono_depth_only_without_stereo or not stereo) and usable[i].all():
                C = cam.position
                r_pred = float(np.linalg.norm(prior[:3] - C))
                r_m = monocular_range(r_pred, predicted[i], observed[i], s.mono_depth_shape_tol)
                ow, oh = observed[i, 1] - observed[i, 0], observed[i, 3] - observed[i, 2]
                pw, ph = predicted[i, 1] - predicted[i, 0], predicted[i, 3] - predicted[i, 2]
                if np.isfinite(r_m):
                    # чувствительность размера рамки к курсу -> в ошибку дальности
                    geom = ObservationGeometry.from_cameras([cam])
                    dh = np.radians(2.0)
                    rot = np.array([np.cos(self.heading + dh), np.sin(self.heading + dh), direction[2]])
                    b2 = self._measure(prior[:3], rot / np.linalg.norm(rot), geom)[0]
                    dsize = abs((b2[1] - b2[0] + b2[3] - b2[2]) - (pw + ph)) / dh
                    rel2 = (s.mono_depth_rel_sigma ** 2 + 2 * self._pixel_sigma2(conf) / (ow + oh) ** 2
                            + (dsize * heading_std / (pw + ph)) ** 2)
                    R = np.array([[rel2 * r_m ** 2]])

                    def fn(x, C=C, r_m=r_m):
                        diff = x[:3] - C
                        r = float(np.linalg.norm(diff))
                        Hm = np.zeros((1, n))
                        Hm[0, :3] = diff / max(r, 1e-9)
                        return np.array([r_m - r]), Hm
                    by_cam.setdefault(d[0], {})["depth"] = (f"depth:{d[0]}", fn, R)
        rejected = 0
        accepted = []
        gates = {1: 6.63, 2: 9.21, 3: 11.34}
        scale = s.aux_gate_chi2 / 9.21
        candidates = list(out) + [m for cam_meas in by_cam.values() for m in cam_meas.values()]
        nis_by_name = {}
        for name, fn, R in candidates:
            r0, H0 = fn(prior)
            S = H0 @ P_prior @ H0.T + R
            try:
                nis = float(r0 @ np.linalg.solve(S, r0))
            except np.linalg.LinAlgError:
                nis = float("inf")
            nis_by_name[name] = nis
            nis_list.append(nis)
            if nis > gates.get(len(r0), 11.34) * scale:
                rejected += 1
                continue
            accepted.append((name, fn, R))
        # конфликт гомографии и дальности одной камеры: дальности по ним расходятся больше 3σ
        names = {a[0] for a in accepted}
        for cid, cam_meas in by_cam.items():
            if "homography" in cam_meas and "depth" in cam_meas and \
                    cam_meas["homography"][0] in names and cam_meas["depth"][0] in names:
                hname, hfn, hR = cam_meas["homography"]
                dname, dfn, dR = cam_meas["depth"]
                C = cameras[cid].position
                xy = hfn(np.zeros(n))[0]                           # y − H·0 = (x, y) гомографии
                p_h = np.r_[xy, prior[2]]
                r_h = float(np.linalg.norm(p_h - C))
                u = (p_h - C)[:2] / max(np.linalg.norm((p_h - C)[:2]), 1e-9)
                var_rh = float(u @ hR @ u)
                r_m = float(dfn(np.r_[C + (p_h - C), np.zeros(n - 3)])[0][0] + r_h)
                if abs(r_h - r_m) > 3 * np.sqrt(var_rh + dR[0, 0]):
                    drop = hname if nis_by_name[hname] > nis_by_name[dname] else dname
                    accepted = [a for a in accepted if a[0] != drop]
                    rejected += 1
        return accepted, rejected, nis_list

    @staticmethod
    def _psd(P: np.ndarray) -> np.ndarray:
        P = (P + P.T) / 2
        w, v = np.linalg.eigh(P)
        if w.min() >= 0:
            return P
        return (v * np.maximum(w, 1e-9)) @ v.T

    # ---------------------------------------------------------------- шаг

    def step(self, frame: int, cameras: dict[int, Camera], detections, dt_frames: float | None = None) -> FrameEstimate:
        """
        Один тик. dt_frames — шаг времени в кадрах (по умолчанию — разность номеров кадров,
        минимум 1); при потоковой обработке передавайте (t − t_prev)·fps.
        Детекция — кортеж (камера, рамка, рёбра в кадре, уверенность, центроид, надёжна[, контур маски]).
        """
        s = self.s
        # «снова увидела» — камеры не было на ПРЕДЫДУЩЕМ ТИКЕ (а не на кадре t − 1): в реальном темпе тики
        # идут через несколько кадров, и иначе каждая камера каждый тик считалась бы только что появившейся
        # (пониженный вес и проверка передачи на всех детекциях)
        prev_tick = self._last_frame if self._last_frame is not None else frame - 1
        for d in detections:
            if self._last_seen.get(d[0], -10 ** 9) < prev_tick:
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
                                     len(detections), 0, 0, -1, dt_frames=dt, observation_level=INITIALIZING)
            prior, P_prior = self.state.copy(), self.P.copy()
            previous_velocity = prior[3:6].copy()
            self._last_stereo = frame
        elif (not self._confirmed or self._bad_streak >= s.reinit_after_bad_frames) \
                and self._try_initialize(frame, cameras, detections, allow_mono=False):
            # стерео стало доступно (после старта по одной камере) или трек разошёлся: состояние
            # заменяется триангуляцией — стерео всегда важнее накопленной гипотезы
            prior, P_prior = self.state.copy(), self.P.copy()
            previous_velocity = prior[3:6].copy()
            self._bad_streak = 0
            self._last_stereo = frame
        else:
            previous_velocity = self.state[3:6].copy()
            F = self._transition(dt)
            prior = F @ self.state
            Q = self._process_noise(dt)
            if s.no_stereo_q_growth_per_s > 0 and self._last_stereo is not None:
                # без стерео прогноз всё менее надёжен: процессный шум растёт со временем без 3D
                seconds = (frame - self._last_stereo) / s.fps
                Q = Q * min(1.0 + s.no_stereo_q_growth_per_s * seconds, s.no_stereo_q_max)
            P_prior = F @ self.P @ F.T + Q

        state, P = prior, P_prior
        occluded_count, used_cameras = 0, 0
        rejected_cameras: list[int] = []
        point = None
        point_used = False
        nis = float("nan")
        reproj_by_camera: dict[int, float] = {}
        direction = self._direction(prior)
        visual, visual_used = None, False
        aux, aux_rejected, aux_nis = [], 0, []
        clip_correction = 0.0
        stereo = False
        any_occluded = False
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
            rejected |= self._handoff_rejects(frame, cameras, detections, P_prior, predicted, observed)
            if rejected.sum() * 2 > len(detections):
                rejected[:] = False
            rejected_cameras = [int(c) for c, r in zip(cam_ids, rejected) if r]
            # закрытые рёбра с гистерезисом: решение удерживается и снимается плавно (вес 0 → 1)
            edge_weight = self._occlusion_hysteresis(frame, detections, detect_occluded_edges(predicted, observed,
                                                                                             valid, s))
            edge_weight[rejected] = 0.0
            occluded = valid & (edge_weight < 1.0) & ~rejected[:, None]
            usable = valid & np.isfinite(predicted) & (edge_weight > 0) & ~rejected[:, None]
            if not s.use_edges:
                usable[:] = False
            occluded_count = int((valid & (edge_weight == 0) & ~rejected[:, None]).sum())
            any_occluded = bool(occluded.any() or rejected.any())
            clean_out = valid.all(axis=1) & ~occluded.any(axis=1) & ~rejected
            if s.occlusion_size_ratio > 0:
                cut = self._size_cut_edges(predicted, observed, valid) & usable
                if cut.any() and ((usable & ~cut).sum(axis=1) >= 3).sum() >= 1:
                    usable = usable & ~cut
                    occluded = occluded | cut
                    occluded_count += int(cut.sum())
            used_cameras = int((usable.sum(axis=1) >= 3).sum())

            if s.visual_heading:
                clean = valid.all(axis=1) & ~occluded.any(axis=1) & ~rejected
                visual, visual_used = self._visual_heading_update(frame, cameras, detections, prior[:3], clean)
                if visual_used:
                    direction = self._direction(prior)
                    predicted = self._measure(prior[:3], direction, geometry)

            silhouette_meas = None
            if s.silhouette_fit and s.silhouette_mode != "output":
                silhouette_meas = self._silhouette_update(cameras, detections, prior, P_prior, valid, occluded,
                                                          rejected)
                if silhouette_meas is not None and silhouette_meas[3]:
                    direction = self._direction(prior)
                    predicted = self._measure(prior[:3], direction, geometry)

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
            stereo = used_cameras >= 2 or point_used
            if s.road_constraint or s.homography or s.mono_depth:
                aux, aux_rejected, aux_nis = self._aux_measurements(prior, P_prior, direction, cameras, detections,
                                                                    predicted, observed, usable, stereo)

            if silhouette_meas is not None and silhouette_meas[0] is not None and s.silhouette_mode == "measure":
                aux = list(aux) + [silhouette_meas[:3]]
            if usable.any() or point_used or aux:
                conf = np.array([max(d[3], 1e-3) for d in detections])
                ramp = np.array([
                    s.reappear_min_weight + (1 - s.reappear_min_weight)
                    * min(1.0, (frame - self._appeared.get(d[0], -10 ** 9)) / max(s.reappear_ramp_frames, 1))
                    for d in detections])
                inflate = np.array([s.edge_inflation_with_point if point_used and c in point.cameras else 1.0
                                    for c in cam_ids])
                for _ in range(1):
                    mask = usable.ravel()
                    observed_flat = observed.ravel()[mask]
                    R_edges = (np.repeat(s.edge_sigma_px / np.sqrt(conf * ramp) * np.sqrt(inflate), 4) ** 2
                               / np.maximum(edge_weight.ravel(), 1e-6))[mask]
                    # систематические источники в квадратуре: несовпадение модели и силуэта, калибровка камеры
                    R_edges = R_edges + s.model_sigma_px ** 2 + s.calibration_sigma_px ** 2
                    if s.heading_in_edges and mask.any():
                        # неопределённость курса: (∂ребро/∂курс)² · σ²_курса
                        dh = np.radians(2.0)
                        rot = np.array([np.cos(self.heading + dh), np.sin(self.heading + dh), direction[2]])
                        rot /= np.linalg.norm(rot)
                        d_edges = (self._measure(prior[:3], rot, geometry) - self._measure(prior[:3], direction, geometry))
                        R_edges = R_edges + (d_edges.ravel()[mask] / dh) ** 2 * self._heading_var_report
                    P_inv = np.linalg.inv(P_prior)
                    measure = lambda x: self._measure(x[:3], direction, geometry).ravel()[mask]  # noqa: E731
                    n_edges = int(mask.sum())
                    H_point = np.zeros((3, self.n))
                    H_point[:, :3] = np.eye(3)
                    R_point_inv = np.linalg.inv(point.covariance) if point_used else None
                    aux_inv = [(fn, np.linalg.inv(R)) for _, fn, R in aux]
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
                        for fn, R_inv in aux_inv:
                            residual_a, H_a = fn(state)
                            information += H_a.T @ R_inv @ H_a
                            gradient += H_a.T @ R_inv @ (residual_a + H_a @ (state - prior))
                        new_state = prior + np.linalg.solve(information, gradient)
                        converged = np.max(np.abs(new_state[:3] - state[:3])) < 1.0
                        state = new_state
                        if converged:
                            break
                    break
                P = np.linalg.inv(information)
                P = (P + P.T) / 2
                unclipped = state
                state = self._apply_limits(prior, state, previous_velocity, dt)
                correction = state - unclipped
                clip_correction = float(np.linalg.norm(correction[:3]))
                if s.clip_covariance and np.any(correction != 0):
                    # ограничение сдвинуло состояние: ковариация не может остаться «как без поправки» —
                    # консервативно добавляем внесённую поправку (P + δδᵀ), положительная полуопределённость
                    P = self._psd(P + np.outer(correction, correction))
            # невязка рёбер после обновления — для планировщика камер и отчёта
            after = self._measure(state[:3], direction, geometry)
            res = np.abs(after - observed)
            for i, c in enumerate(cam_ids):
                m = usable[i]
                reproj_by_camera[int(c)] = float(res[i][m].mean()) if m.any() else float("nan")

            if not stereo and s.depth_sigma_growth_mps > 0 and self._last_stereo is not None and used_cameras:
                # одна камера: глубина вдоль её луча не наблюдается — СКО вдоль луча растёт с временем без стерео
                best = int(np.argmax(usable.sum(axis=1)))
                ray = state[:3] - cameras[cam_ids[best]].position
                ray /= max(np.linalg.norm(ray), 1e-9)
                t1 = (frame - self._last_stereo) / s.fps
                t0 = max(t1 - dt / s.fps, 0.0)
                rate = s.depth_sigma_growth_mps * s.units_per_meter
                P = P.copy()
                P[:3, :3] += ((rate * t1) ** 2 - (rate * t0) ** 2) * np.outer(ray, ray)

        self.state, self.P = state, P
        bad = (point is not None and point.ok and not point_used) or len(rejected_cameras) >= 2
        self._bad_streak = self._bad_streak + 1 if bad else 0
        if used_cameras >= 2 or point_used:
            self._confirmed = True
            self._last_confirmed = frame
        if stereo:
            self._last_stereo = frame
            if self.road is not None and s.road_mode != "fixed" and used_cameras >= 2:
                self.road.add(state[:3] - np.array([0.0, 0.0, float(self.dims[2]) / 2]))
        status = MEASURED if (used_cameras or point_used or aux) else (PARTIAL if detections else PREDICTED)
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
        if s.heading_output_rate_limit and self._out_heading is not None:
            speed_ms = self._speed_mps(state)
            rate = speed_ms / max(s.min_turn_radius_m, 0.1) + np.radians(s.heading_rate_floor_dps)
            lim = rate / s.fps * max(dt, 1e-6)
            delta = float(_wrap(self.heading - self._out_heading))
            if s.heading_accel_limit_dps2 > 0:
                dw = np.radians(s.heading_accel_limit_dps2) / s.fps ** 2 * max(dt, 1e-6) ** 2
                delta = float(np.clip(delta, self._out_yaw_rate * dt - dw, self._out_yaw_rate * dt + dw))
            delta = float(np.clip(delta, -lim, lim))
            self._out_yaw_rate = delta / max(dt, 1e-6)
            if delta != float(_wrap(self.heading - self._out_heading)):
                self.heading = float(_wrap(self._out_heading + delta))
                d = np.array([np.cos(self.heading), np.sin(self.heading), self.heading_dir[2]
                              / max(np.hypot(*self.heading_dir[:2]), 1e-9)])
                self.heading_dir = d / np.linalg.norm(d)
                direction = self.heading_dir
        self._out_heading = self.heading
        out_center, out_heading_cov = state[:3], None
        if s.silhouette_fit and s.silhouette_mode == "output" and detections:
            masks = {d[0]: d[6] for d, ok in zip(detections, clean_out) if ok and len(d) > 6 and d[6] is not None}
            if len(masks) >= self._sf.min_views:
                slope = float(direction[2] / max(np.hypot(*direction[:2]), 1e-9))
                views = select_views(cameras, masks, state[:3], self.heading, slope, self.model, self.dims, self._sf)
                fit = fit_pose(views, state[:3], self.heading, slope, self.model, self.dims, self._sf)
                if fit.ok:
                    out_center = np.array([fit.x, fit.y, state[2]])
                    d = np.array([np.cos(fit.heading), np.sin(fit.heading), slope])
                    direction = d / np.linalg.norm(d)
                    out_heading_cov = fit.covariance
        forward, side, up = truck_axes(direction[None, :])
        if self.model is not None:
            points = self.model.to_world(out_center, forward[0], side[0], up[0],
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
        # уровень наблюдения: что реально измерено на этом кадре
        if stereo:
            level = "multi_partial" if any_occluded else "multi"
        elif used_cameras == 1:
            level = "single"
        elif aux:
            level = "road"
        else:
            level = "lost" if since >= s.lost_after_frames else "predicted"
        # ковариация точки вывода: центр + рычаг (точка вывода смещена от центра) × неопределённость курса
        lever = position - state[:3]
        j = np.array([-lever[1], lever[0], 0.0])
        ref_cov = P[:3, :3] + self._heading_var_report * np.outer(j, j)
        if s.model_length_sigma > 0:
            # точка вывода на краю модели: ошибка длины сдвигает её вдоль оси на половину ошибки
            ref_cov = ref_cov + (s.model_length_sigma / 2) ** 2 * np.outer(forward[0], forward[0])
        return FrameEstimate(
            frame, status, position, state[:3].copy(), cab_center, speed, float(self.heading),
            float(np.sqrt(P[0, 0] + P[1, 1])), len(detections), used_cameras, occluded_count, since,
            rejected_cameras=rejected_cameras, velocity_mps=state[3:6] * to_mps, acceleration_mps2=accel_mps2,
            accel_magnitude_mps2=float(np.linalg.norm(accel_mps2)), covariance=P[:3, :3].copy(),
            point_used=point_used, point_cameras=list(point.cameras) if point_used else [],
            point_nis=nis, point_std_m=point.std_m if point is not None else float("nan"),
            reproj_px=float(np.mean(reproj_vals)) if reproj_vals else float("nan"),
            reproj_by_camera=reproj_by_camera, dt_frames=dt, quality=quality,
            observation_level=level, reference_covariance=ref_cov, heading_std=float(np.sqrt(self._heading_var_report)),
            visual_heading=float(visual.heading) if visual is not None and visual.ok else float("nan"),
            visual_heading_used=visual_used, aux_used=[a[0] for a in aux], aux_rejected=aux_rejected,
            aux_nis=aux_nis, frames_since_stereo=-1 if self._last_stereo is None else frame - self._last_stereo,
            clip_correction=clip_correction,
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
