"""
Триангуляция в реальном времени и формирование 3D-измерения для фильтра.

Задачи модуля (ТЗ 3.2):
1. Триангуляция по детекциям текущего тика: робастная (перебор подмножеств камер,
   `triangulation.triangulate_robust`), с проверками геометрии — ошибка перепроецирования,
   точка перед всеми камерами, угол пересечения лучей (почти параллельные лучи не дают
   глубины), — и с ковариацией 3D-точки из якобиана проекции.
2. Согласование физических точек. Центроид маски на разных ракурсах — разные точки кузова,
   поэтому «сырые» центроиды дают смещённую точку. Режим `model_corrected`: по прогнозу
   фильтра для каждой камеры считается, куда проецируется ЦЕНТР 3D-модели и где лежит центроид
   её силуэта; разность — поправка ракурса; центроид маски минус поправка ≈ пиксель центра
   модели. Триангуляция таких пикселей даёт точку, согласованную между камерами, и измерение
   относится к той же точке, что и состояние фильтра. Это не «выдумывание» ключевой точки:
   используется только геометрия известной модели и прогноз; ошибка поправки входит в
   ковариацию через `covariance_inflation`.
3. Измерение для фильтра: точка + ковариация + список камер; если точность по геометрии
   недостаточна (угол лучей, СКО) — измерение не выдаётся, фильтр работает по рёбрам/прогнозу
   и помечает снижение качества.

Ковариация точки: для камеры i при глубине z и координатах камеры (x, y, z)
    J_i = ∂(u, v)/∂X = (1/z) · [f_x (r₁ − (x/z) r₃); f_y (r₂ − (y/z) r₃)] (2×3, r_k — строки R),
    Σ = (Σ_i J_iᵀ J_i / σ_px²)⁻¹ — обратная информационная матрица Гаусса–Ньютона.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from reconstruction.cab_model import box_corners, truck_axes
from reconstruction.camera import Camera
from reconstruction.triangulation import triangulate_robust


@dataclass
class FusionSettings:
    mode: str = "model_corrected"          # none | centroid | model_corrected
    sigma_px: float = 8.0
    max_reproj_px: float = 20.0
    min_ray_angle_deg: float = 2.0
    max_point_std_m: float = 5.0
    covariance_inflation: float = 1.5
    units_per_meter: float = 100.0


@dataclass
class PointMeasurement:
    point: np.ndarray                                  # (3,) ед. мира; NaN, если ok = False
    covariance: np.ndarray                             # (3, 3)
    cameras: list[int] = field(default_factory=list)   # камеры-инлаеры
    reprojection_px: dict = field(default_factory=dict)
    min_ray_angle_deg: float = float("nan")
    ok: bool = False
    reason: str = ""
    std_m: float = float("nan")                        # СКО точки (корень из средней дисперсии), м


def projection_jacobian(camera: Camera, point: np.ndarray) -> np.ndarray:
    """∂(u, v)/∂X (2×3) для точки перед камерой."""
    cam = camera.R @ (np.asarray(point, float) - camera.position)
    x, y, z = cam
    z = max(z, 1e-6)
    r1, r2, r3 = camera.R
    return np.vstack([camera.f_x * (r1 - x / z * r3), camera.f_y * (r2 - y / z * r3)]) / z


def point_covariance(cameras: list[Camera], point: np.ndarray, sigma_px: float) -> np.ndarray:
    info = np.zeros((3, 3))
    for cam in cameras:
        J = projection_jacobian(cam, point)
        info += J.T @ J
    info /= sigma_px ** 2
    try:
        return np.linalg.inv(info + np.eye(3) * 1e-12)
    except np.linalg.LinAlgError:
        return np.full((3, 3), np.inf)


def ray_angles_deg(cameras: list[Camera], point: np.ndarray) -> np.ndarray:
    """Углы между лучами «камера → точка» для всех пар, градусы."""
    rays = [(np.asarray(point, float) - c.position) for c in cameras]
    rays = [r / max(np.linalg.norm(r), 1e-9) for r in rays]
    out = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            out.append(np.degrees(np.arccos(np.clip(abs(rays[i] @ rays[j]), -1.0, 1.0))))
    return np.array(out)


def triangulate_realtime(cameras: list[Camera], camera_ids: list[int], pixels: list[tuple[float, float]],
                         settings: FusionSettings) -> PointMeasurement:
    """Точка по детекциям одного тика с проверками геометрии и ковариацией."""
    nan = np.full(3, np.nan)
    big = np.diag([np.inf, np.inf, np.inf])
    if len(cameras) < 2:
        return PointMeasurement(nan, big, [], {}, reason="меньше двух камер")
    res = triangulate_robust(cameras, pixels, max_reprojection_error_px=settings.max_reproj_px)
    reproj = {cid: float(e) for cid, e in zip(camera_ids, res.reprojection_errors)}
    if not res.ok:
        return PointMeasurement(nan, big, [], reproj, reason=res.reason or "триангуляция не удалась")
    inl = [i for i, keep in enumerate(res.inliers) if keep]
    cams = [cameras[i] for i in inl]
    ids = [camera_ids[i] for i in inl]
    if any(c.depth(res.point) <= 0 for c in cams):
        return PointMeasurement(nan, big, ids, reproj, reason="точка позади камеры")
    angles = ray_angles_deg(cams, res.point)
    min_angle = float(angles.min()) if len(angles) else float("nan")
    if len(angles) and angles.max() < settings.min_ray_angle_deg:
        return PointMeasurement(res.point, big, ids, reproj, min_angle, False, "лучи почти параллельны")
    cov = point_covariance(cams, res.point, settings.sigma_px) * settings.covariance_inflation
    std_m = float(np.sqrt(np.trace(cov) / 3)) / settings.units_per_meter
    if not np.isfinite(std_m) or std_m > settings.max_point_std_m:
        return PointMeasurement(res.point, cov, ids, reproj, min_angle, False, f"СКО точки {std_m:.1f} м слишком велико",
                                std_m)
    return PointMeasurement(res.point, cov, ids, reproj, min_angle, True, "", std_m)


def _polygon_centroid(points_2d: np.ndarray) -> np.ndarray:
    """Центроид площади выпуклой оболочки точек (K, 2); при вырождении — среднее точек."""
    if len(points_2d) < 3:
        return points_2d.mean(axis=0)
    try:
        hull = ConvexHull(points_2d)
    except Exception:  # noqa: BLE001 — вырожденная фигура
        return points_2d.mean(axis=0)
    poly = points_2d[hull.vertices]
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        return poly.mean(axis=0)
    return np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / (6 * area)


def silhouette_offsets(direction: np.ndarray, model=None, dims: np.ndarray | None = None) -> np.ndarray:
    """Точки формы грузовика относительно центра (K, 3) в мире при курсе direction (считать один раз на тик)."""
    forward, side, up = (a[0] for a in truck_axes(np.asarray(direction, float)[None, :]))
    if model is not None:
        return model.to_world(np.zeros(3), forward, side, up, model.hull)
    dims = np.asarray(dims, float)
    reference = dims[0] / 2 * forward - dims[2] / 2 * up
    return box_corners(reference[None, :], np.asarray(direction, float)[None, :], dims)[0]


def silhouette_points(center: np.ndarray, direction: np.ndarray, model=None, dims: np.ndarray | None = None,
                      offsets: np.ndarray | None = None) -> np.ndarray:
    """Точки формы грузовика в мире (K, 3): оболочка модели или углы коробки с центром center."""
    if offsets is None:
        offsets = silhouette_offsets(direction, model, dims)
    return np.asarray(center, float) + offsets


def centroid_offset(camera: Camera, center: np.ndarray, direction: np.ndarray, model=None,
                    dims: np.ndarray | None = None) -> np.ndarray | None:
    """
    Поправка ракурса: (центроид силуэта модели − проекция центра модели) в пикселях при прогнозном
    положении. None, если модель не перед камерой.
    """
    pts = silhouette_points(center, direction, model, dims)
    uv = camera.project(pts)
    c = camera.project(np.asarray(center, float))
    if not np.all(np.isfinite(uv)) or not np.all(np.isfinite(c)):
        return None
    return _polygon_centroid(uv) - c


def build_point_measurement(cameras: dict[int, Camera], centroids: dict[int, tuple[float, float]],
                            settings: FusionSettings, predicted_center: np.ndarray | None = None,
                            direction: np.ndarray | None = None, model=None,
                            dims: np.ndarray | None = None) -> PointMeasurement:
    """
    3D-измерение центра объекта по центроидам масок активных камер.

    mode = centroid: триангуляция сырых центроидов (смещена: разные точки кузова).
    mode = model_corrected: центроиды исправляются поправкой ракурса по прогнозу (нужны
    predicted_center и direction; без них — как centroid).
    """
    nan = np.full(3, np.nan)
    if settings.mode == "none" or len(centroids) < 2:
        return PointMeasurement(nan, np.diag([np.inf] * 3), [], {}, reason="режим none" if settings.mode == "none"
                                else "меньше двух центроидов")
    ids = sorted(centroids)
    pixels = []
    for cid in ids:
        uv = np.asarray(centroids[cid], float)
        if settings.mode == "model_corrected" and predicted_center is not None and direction is not None:
            off = centroid_offset(cameras[cid], predicted_center, direction, model, dims)
            if off is not None:
                uv = uv - off
        pixels.append((float(uv[0]), float(uv[1])))
    return triangulate_realtime([cameras[c] for c in ids], ids, pixels, settings)
