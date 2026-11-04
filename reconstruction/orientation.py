"""
Курс грузовика непосредственно по изображению (независимо от направления скорости).

Признаки:
1. Главная ось маски (PCA по моментам площади контура) — ось в изображении, неоднозначность 180°;
   `mask_axis` (диагностика и тесты; в 3D-курс сама по себе не переводится — ось на изображении
   зависит от ракурса).
2. Сопоставление силуэта: для курса θ выпуклая оболочка 3D-модели при прогнозном центре
   проецируется в камеру, совмещается с маской по центроиду площади (ошибка положения центра не
   смещает оценку курса) и сравнивается с выпуклой оболочкой маски по IoU (растр 64×64).
   S_c(θ) — кривая по кругу курсов для камеры c.
3. Согласование камер: S(θ) = Σ_c w_c · S_c(θ) (w_c — уверенность детектора). Камера с обрезанной
   краем кадра или закрытой рамкой, маленькой маской или плохим лучшим совпадением (IoU < min_iou —
   ошибка сегментации или несовпадение модели) не участвует.

Измерение отдельно оценивает:
- достоверность ОСИ — контраст сложенной по модулю 180° кривой (лучшая ось против худшей);
- достоверность направления «перед/зад» — разность S(θ*) − S(θ* + 180°). Если она меньше порога,
  выдаётся только ось (axis_only = True): трекер выберет из двух направлений ближайшее к своему курсу,
  а не «придумает» перед.
СКО курса — полуширина пика кривой на уровне peak − score_drop (не меньше min_sigma_deg).
Временная согласованность и вес измерения против прогноза — в трекере (visibility.py).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from reconstruction.camera import Camera
from reconstruction.fusion import silhouette_offsets

CANVAS = 64


@dataclass
class OrientationSettings:
    step_deg: float = 10.0               # грубая сетка курсов
    refine_deg: float = 2.0              # уточнение вокруг пиков
    min_iou: float = 0.55                # лучшее совпадение камеры ниже — камера не используется
    min_mask_px: float = 40.0            # меньший размер рамки маски — не используется
    min_axis_contrast: float = 0.04      # контраст оси (сумма по камерам на камеру) ниже — измерения нет
    min_front_margin: float = 0.03       # перевес «перед/зад» (на камеру) ниже — только ось
    score_drop: float = 0.02             # уровень для полуширины пика
    min_sigma_deg: float = 4.0
    max_sigma_deg: float = 45.0


@dataclass
class HeadingMeasurement:
    heading: float = float("nan")        # радианы (ось X → Y)
    sigma: float = float("nan")          # СКО, радианы
    axis_only: bool = True               # True — направление «перед/зад» не определено (±180°)
    axis_contrast: float = 0.0
    front_margin: float = 0.0
    cameras: list = field(default_factory=list)
    ok: bool = False
    reason: str = ""


def polygon_moments(polygon: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Площадь, центроид (2,) и ковариация площади (2, 2) многоугольника (формулы Грина)."""
    p = np.asarray(polygon, float)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        c = p.mean(axis=0)
        return 0.0, c, np.cov(p.T) if len(p) > 1 else np.zeros((2, 2))
    cx = ((x + x1) * cross).sum() / (6 * area)
    cy = ((y + y1) * cross).sum() / (6 * area)
    sxx = ((x * x + x * x1 + x1 * x1) * cross).sum() / (12 * area) - cx * cx
    syy = ((y * y + y * y1 + y1 * y1) * cross).sum() / (12 * area) - cy * cy
    sxy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cross).sum() / (24 * area) - cx * cy
    return abs(area), np.array([cx, cy]), np.array([[sxx, sxy], [sxy, syy]])


def mask_axis(polygon: np.ndarray) -> tuple[float, float]:
    """Главная ось маски в изображении: угол (радианы, mod π) и вытянутость (λ1/λ2 ≥ 1)."""
    _, _, cov = polygon_moments(polygon)
    w, v = np.linalg.eigh(cov)
    angle = float(np.arctan2(v[1, 1], v[0, 1])) % np.pi
    return angle, float(np.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12)))


def _raster(points: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    img = np.zeros((CANVAS, CANVAS), np.uint8)
    q = np.round((points - origin) * scale + CANVAS / 2).astype(np.int32)
    hull = cv2.convexHull(q)
    cv2.fillConvexPoly(img, hull, 1)
    return img


class SilhouetteMatcher:
    """Кривая совпадения силуэта по курсам для одной камеры (модель или коробка)."""

    def __init__(self, settings: OrientationSettings | None = None, model=None, dims: np.ndarray | None = None):
        self.s = settings or OrientationSettings()
        self.model = model
        self.dims = None if dims is None else np.asarray(dims, float)
        self._offsets: dict[float, np.ndarray] = {}

    slope: float = 0.0                   # продольный уклон (dz / горизонталь) — грузовик на склоне

    def offsets(self, heading: float) -> np.ndarray:
        key = (round(float(heading) % (2 * np.pi), 6), round(self.slope, 2))
        if key not in self._offsets:
            if len(self._offsets) > 1024:          # кеш ограничен
                self._offsets.clear()
            d = np.array([np.cos(key[0]), np.sin(key[0]), key[1]])
            self._offsets[key] = silhouette_offsets(d / np.linalg.norm(d), self.model, self.dims)
        return self._offsets[key]

    def scores(self, camera: Camera, polygon: np.ndarray, center: np.ndarray, headings: np.ndarray) -> np.ndarray:
        """IoU выпуклых оболочек (маска против проекции модели, совмещение по центроиду) для каждого курса."""
        poly = np.asarray(polygon, float)
        area, centroid, _ = polygon_moments(cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2))
        span = np.ptp(poly, axis=0).max()
        out = np.full(len(headings), np.nan)
        if area <= 0 or span <= 0:
            return out
        scale = 0.6 * CANVAS / span
        mask_img = _raster(poly, centroid, scale)
        mask_n = int(mask_img.sum())
        for i, h in enumerate(headings):
            uv = camera.project(np.asarray(center, float) + self.offsets(h))
            if not np.all(np.isfinite(uv)):
                continue
            hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
            _, c, _ = polygon_moments(hull)
            img = _raster(hull, c, scale)
            inter = int(np.count_nonzero(img & mask_img))
            out[i] = inter / max(mask_n + int(img.sum()) - inter, 1)
        return out


def _wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def estimate_heading(matcher: SilhouetteMatcher, views: list[tuple[int, Camera, np.ndarray, float]],
                     center: np.ndarray) -> HeadingMeasurement:
    """
    views — [(камера, Camera, контур маски (K, 2), вес)] пригодных камер. Возвращает измерение курса
    с раздельной достоверностью оси и направления.
    """
    s = matcher.s
    grid = np.radians(np.arange(0.0, 360.0, s.step_deg))
    total = np.zeros(len(grid))
    used, curves, weights = [], [], []
    for cid, cam, poly, w in views:
        if poly is None or len(poly) < 3 or np.ptp(np.asarray(poly), axis=0).min() < s.min_mask_px:
            continue
        sc = matcher.scores(cam, poly, center, grid)
        if not np.all(np.isfinite(sc)) or np.nanmax(sc) < s.min_iou:
            continue                                       # маска не похожа на модель ни при каком курсе
        used.append(cid)
        curves.append((cam, poly, w))
        weights.append(w)
        total += w * sc
    if not used:
        return HeadingMeasurement(reason="нет пригодных масок")
    wsum = float(np.sum(weights))
    best = int(np.argmax(total))

    def refine(center_angle: float) -> tuple[float, float, np.ndarray, np.ndarray]:
        local = center_angle + np.radians(np.arange(-s.step_deg, s.step_deg + 1e-9, s.refine_deg))
        val = sum(w * matcher.scores(cam, poly, center, local) for cam, poly, w in curves)
        k = int(np.nanargmax(val))
        return float(local[k]), float(val[k]), local, val

    h1, v1, local1, val1 = refine(grid[best])
    _, v2, _, _ = refine(h1 + np.pi)
    folded = np.maximum(total, np.roll(total, len(grid) // 2))
    axis_contrast = (max(v1, v2) - float(folded.min())) / wsum
    front_margin = (v1 - v2) / wsum
    if axis_contrast < s.min_axis_contrast:
        return HeadingMeasurement(float(h1), float("nan"), True, axis_contrast, front_margin, used, False,
                                  "ось не определена (силуэт почти не зависит от курса)")
    # полуширина пика: по грубой кривой вокруг лучшего направления (на уровне peak − score_drop · wsum)
    level = max(v1, float(total[best])) - s.score_drop * wsum
    above = np.r_[val1 >= level]
    half = max(np.ptp(local1[above]) / 2 if above.any() else 0.0, 0.0)
    ang = _wrap(grid - h1)
    near = np.abs(ang) <= np.pi / 2
    wide = np.abs(ang[near & (total >= level)])
    half = max(half, float(wide.max()) if len(wide) else 0.0)
    sigma = float(np.clip(half, np.radians(s.min_sigma_deg), np.radians(s.max_sigma_deg)))
    return HeadingMeasurement(float(_wrap(h1)), sigma, front_margin < s.min_front_margin, axis_contrast,
                              front_margin, used, True)
