"""
Уточнение положения и курса подгонкой 3D-модели ко ВСЕМУ контуру масок нескольких камер.

Рамка детектора даёт 4 числа на камеру, и курс по ней определяется плохо (рамка почти не меняется при
повороте). Контур маски несёт форму: для параметров (x, y, курс) — z центра берётся из прогноза —
выпуклая оболочка модели проецируется в каждую камеру и сравнивается с выпуклой оболочкой маски по
симметричному расстоянию контуров (точки контура проекции → контур маски и наоборот, px). Подгонка —
least_squares с Хьюбером от прогноза трека, со слабой привязкой к прогнозу (стабилизация).

Используются только «чистые» камеры: рамка не касается края кадра, ребра не закрыты, и площадь маски
согласуется с площадью проекции модели в прогнозе (иначе маска частично закрыта препятствием — такая
маска тянет модель в сторону). Нужно ≥ min_views камер: по одной камере подгонка на реальных данных
хуже прогноза (глубина не определена).

Ковариация результата — (JᵀJ)⁻¹ · σ²_невязки (σ по медиане невязок, не меньше sigma_floor_px), с запасом
inflation: измерение коррелировано с рёбрами рамок тех же камер.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from reconstruction.fusion import silhouette_offsets


@dataclass
class SilhouetteFitSettings:
    min_views: int = 2
    area_ratio_min: float = 0.7          # площадь маски / площадь проекции модели в прогнозе ∈ [r, 1/r]
    area_heading_tol_deg: float = 0.0    # сравнивать с лучшим из курсов ±tol (0 — только курс прогноза; ±15°
                                         # на реальном видео пропускает больше плохих масок: макс 8.7 → 11.7 м)
    samples: int = 48                    # точек на контур
    huber_px: float = 6.0
    prior_sigma: float = 50.0            # ед. мира — слабая привязка к прогнозу
    prior_sigma_heading_deg: float = 5.0
    sigma_floor_px: float = 3.0
    inflation: float = 2.0
    max_nfev: int = 40


@dataclass
class SilhouetteFit:
    ok: bool
    x: float = float("nan")
    y: float = float("nan")
    heading: float = float("nan")
    covariance: np.ndarray | None = None        # (3, 3): x, y (ед. мира²), курс (рад²)
    cameras: list | None = None
    rms_px: float = float("nan")
    reason: str = ""


def _sample(poly: np.ndarray, n: int) -> np.ndarray:
    p = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    i = np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(seg) - 1)
    a = (t - s[i]) / np.maximum(seg[i], 1e-9)
    return p[i] + (p[i + 1] - p[i]) * a[:, None]


def _dist_to_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    ap = pts[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9), 0.0, 1.0)
    return np.linalg.norm(pts[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(axis=1)


def _hull(uv: np.ndarray) -> np.ndarray:
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(float)


def select_views(cameras: dict, masks: dict, center: np.ndarray, heading: float, slope: float, model, dims,
                 s: SilhouetteFitSettings) -> list:
    """Камеры с чистой маской: площадь маски согласуется с проекцией модели в прогнозе."""
    # площадь проекции при виде почти с торца сильно зависит от курса: сравниваем с лучшим из курсов ±15°,
    # чтобы ошибка курса прогноза не отбрасывала хорошую маску (а закрытая наполовину — по-прежнему не проходит)
    shapes = []
    tol = s.area_heading_tol_deg
    for dh in np.radians([-tol, -tol / 2, 0.0, tol / 2, tol] if tol > 0 else [0.0]):
        d = np.array([np.cos(heading + dh), np.sin(heading + dh), slope])
        shapes.append(center + silhouette_offsets(d / np.linalg.norm(d), model, dims))
    views = []
    for cid, poly in masks.items():
        if poly is None or len(poly) < 5:
            continue
        cam = cameras[cid]
        mh = _hull(np.asarray(poly, float))
        area = cv2.contourArea(mh.astype(np.float32))
        best = 0.0
        for pts in shapes:
            uv = cam.project(pts)
            if np.all(np.isfinite(uv)):
                r = area / max(cv2.contourArea(_hull(uv).astype(np.float32)), 1.0)
                best = max(best, min(r, 1.0 / max(r, 1e-9)))
        if best >= s.area_ratio_min:
            views.append((cid, cam, mh, _sample(mh, s.samples)))
    return views


def fit_pose(views: list, center: np.ndarray, heading: float, slope: float, model, dims,
             s: SilhouetteFitSettings) -> SilhouetteFit:
    if len(views) < s.min_views:
        return SilhouetteFit(False, reason=f"чистых масок {len(views)} < {s.min_views}")
    z = float(center[2])
    prior = np.array([center[0], center[1], heading])
    sig_h = np.radians(s.prior_sigma_heading_deg)

    def residuals(p):
        x, y, h = p
        d = np.array([np.cos(h), np.sin(h), slope])
        pts = np.array([x, y, z]) + silhouette_offsets(d / np.linalg.norm(d), model, dims)
        out = []
        for _, cam, mh, ms in views:
            uv = cam.project(pts)
            if not np.all(np.isfinite(uv)):
                out.append(np.full(2 * s.samples, 50.0))
                continue
            ph = _hull(uv)
            out.append(_dist_to_poly(_sample(ph, s.samples), mh))
            out.append(_dist_to_poly(ms, ph))
        out.append([(x - prior[0]) / s.prior_sigma * s.huber_px, (y - prior[1]) / s.prior_sigma * s.huber_px,
                    (h - prior[2]) / sig_h * s.huber_px])
        return np.concatenate(out)

    try:
        sol = least_squares(residuals, prior, loss="huber", f_scale=s.huber_px, x_scale=[100.0, 100.0, 0.05],
                            diff_step=[1e-3, 1e-3, 1e-2], max_nfev=s.max_nfev)
    except Exception as e:  # noqa: BLE001
        return SilhouetteFit(False, reason=f"подгонка не удалась: {e}")
    r = sol.fun[:-3]
    sigma = max(float(np.median(np.abs(r))) * 1.4826, s.sigma_floor_px)
    J = sol.jac[:-3]
    try:
        cov = np.linalg.inv(J.T @ J + np.diag([1e-12, 1e-12, 1e-12])) * sigma ** 2 * s.inflation
    except np.linalg.LinAlgError:
        return SilhouetteFit(False, reason="вырожденная подгонка")
    if not np.all(np.isfinite(cov)):
        return SilhouetteFit(False, reason="вырожденная подгонка")
    return SilhouetteFit(True, float(sol.x[0]), float(sol.x[1]), float(sol.x[2]), cov,
                         [v[0] for v in views], float(np.sqrt(np.mean(r ** 2))))
