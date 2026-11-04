"""
Дорожная плоскость и гомография «дорога ↔ изображение» для каждой камеры.

Соглашения проекта (reconstruction/camera.py): R — поворот мир → камера, строки R — оси камеры
(right, −up, forward); пиксель p ~ K · R · (X − C), K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]].
Точка плоскости X = X0 + a·e1 + b·e2 (X0 — точка плоскости, e1, e2 — базис), тогда
    R (X − C) = a·R e1 + b·R e2 + R (X0 − C)   ⇒   H = K [R e1 | R e2 | R (X0 − C)],
p ~ H · (a, b, 1). Формула совпадает с общей для R «мир → камера»; проверена тестами
(прямое и обратное преобразование против Camera.project).

Гомография верна ТОЛЬКО для точек на этой плоскости. Центроид маски грузовика лежит над дорогой,
поэтому в дорожную плоскость переводится точка основания (низ силуэта), а не центр маски.

Плоскость дороги:
- RoadPlane — одна плоскость (точка и нормаль); `horizontal(z)` — горизонтальная на высоте z;
- OnlineRoadModel — кусочно-плоская модель, оцениваемая причинно по собственным уверенным
  оценкам трека (кадры с ≥2 камерами): точка контакта = центр модели − H/2 по вертикали.
  Локальная плоскость z = a + b·x + c·y по точкам в радиусе от текущего положения, СКО — по разбросу
  невязок (неопределённость растёт там, где точек мало или дорога неровная). Буфер ограничен.
  CSV цели не используется.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from reconstruction.camera import Camera


@dataclass
class RoadPlane:
    point: np.ndarray                    # (3,) X0, ед. мира
    normal: np.ndarray                   # (3,) единичная нормаль (вверх)
    sigma: float = 0.0                   # СКО высоты плоскости, ед. мира

    def __post_init__(self):
        self.point = np.asarray(self.point, float)
        n = np.asarray(self.normal, float)
        self.normal = n / max(np.linalg.norm(n), 1e-12)
        if self.normal[2] < 0:
            self.normal = -self.normal

    @classmethod
    def horizontal(cls, z: float, sigma: float = 0.0) -> "RoadPlane":
        return cls(np.array([0.0, 0.0, z]), np.array([0.0, 0.0, 1.0]), sigma)

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """e1, e2 — ортонормированный базис плоскости (e1 — проекция оси X мира)."""
        n = self.normal
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = ref - (ref @ n) * n
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(n, e1)

    def height_at(self, xy: np.ndarray) -> float:
        """z плоскости над точкой (x, y)."""
        n, p = self.normal, self.point
        if abs(n[2]) < 1e-9:
            return float("nan")
        return float(p[2] - (n[0] * (xy[0] - p[0]) + n[1] * (xy[1] - p[1])) / n[2])

    def slope(self) -> np.ndarray:
        """(∂z/∂x, ∂z/∂y)."""
        n = self.normal
        return np.array([-n[0] / n[2], -n[1] / n[2]])

    def to_plane(self, X: np.ndarray) -> np.ndarray:
        e1, e2 = self.basis
        d = np.asarray(X, float) - self.point
        return np.stack([d @ e1, d @ e2], axis=-1)

    def from_plane(self, ab: np.ndarray) -> np.ndarray:
        e1, e2 = self.basis
        ab = np.asarray(ab, float)
        return self.point + ab[..., 0:1] * e1 + ab[..., 1:2] * e2


@dataclass
class Homography:
    H: np.ndarray                        # (3, 3) плоскость (a, b, 1) -> пиксель
    H_inv: np.ndarray
    plane: RoadPlane
    camera: Camera
    condition: float                     # число обусловленности H (с нормировкой масштаба)
    ok: bool
    reason: str = ""


def ground_homography(camera: Camera, plane: RoadPlane, max_condition: float = 1e12) -> Homography:
    """H = K [R e1 | R e2 | R (X0 − C)]; проверка вырожденности (камера в плоскости, плоскость через центр камеры)."""
    e1, e2 = plane.basis
    R = camera.R
    H = camera.K @ np.c_[R @ e1, R @ e2, R @ (plane.point - camera.position)]
    # масштаб столбцов разный (пиксели и ед. мира) — для обусловленности нормируем
    scale = np.diag(1.0 / np.maximum(np.linalg.norm(H, axis=0), 1e-12))
    try:
        cond = float(np.linalg.cond(H @ scale))
    except np.linalg.LinAlgError:
        cond = float("inf")
    height = float((camera.position - plane.point) @ plane.normal)
    if not np.isfinite(cond) or cond > max_condition:
        return Homography(H, np.full((3, 3), np.nan), plane, camera, cond, False, "гомография вырождена")
    if abs(height) < 1e-6:
        return Homography(H, np.full((3, 3), np.nan), plane, camera, cond, False, "камера в плоскости дороги")
    return Homography(H, np.linalg.inv(H), plane, camera, cond, True)


def plane_to_pixel(h: Homography, ab: np.ndarray) -> np.ndarray:
    """Координаты плоскости (N, 2) -> пиксели (N, 2); точки позади камеры — NaN."""
    ab = np.atleast_2d(np.asarray(ab, float))
    p = np.c_[ab, np.ones(len(ab))] @ h.H.T
    out = p[:, :2] / p[:, 2:3]
    out[p[:, 2] <= 0] = np.nan
    return out


def pixel_to_plane(h: Homography, uv: np.ndarray, max_range: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Пиксели (N, 2) -> координаты плоскости (N, 2) и маска допустимости. Недопустимы: пиксель выше
    горизонта (луч не пересекает плоскость перед камерой), дальше max_range от камеры (рабочая область:
    у горизонта малая ошибка пикселя — огромная ошибка на дороге).
    """
    uv = np.atleast_2d(np.asarray(uv, float))
    if not h.ok:
        return np.full((len(uv), 2), np.nan), np.zeros(len(uv), bool)
    q = np.c_[uv, np.ones(len(uv))] @ h.H_inv.T
    with np.errstate(divide="ignore", invalid="ignore"):
        ab = q[:, :2] / q[:, 2:3]
    # точка перед камерой: глубина её проекции положительна
    X = h.plane.from_plane(ab)
    depth = (X - h.camera.position) @ h.camera.forward
    ok = np.isfinite(ab).all(axis=1) & (depth > 0)
    if max_range is not None:
        ok &= np.linalg.norm(X - h.camera.position, axis=1) <= max_range
    ab[~ok] = np.nan
    return ab, ok


def pixel_to_ground(h: Homography, uv: np.ndarray, max_range: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Пиксели -> мировые точки на плоскости (N, 3) и маска допустимости."""
    ab, ok = pixel_to_plane(h, uv, max_range)
    return h.plane.from_plane(ab), ok


def ground_jacobian(h: Homography, uv: np.ndarray) -> np.ndarray:
    """∂X/∂(u, v) (3×2) для точки плоскости, видимой в пикселе uv (аналитически через H⁻¹)."""
    q = h.H_inv @ np.r_[np.asarray(uv, float), 1.0]
    a, b, w = q
    # ab = (a/w, b/w); ∂ab/∂uv = (1/w) [I | −ab] · H_inv[:, :2]
    J_q = h.H_inv[:, :2]
    J_ab = (J_q[:2] - np.outer([a / w, b / w], J_q[2])) / w
    e1, e2 = h.plane.basis
    return np.c_[e1, e2] @ J_ab


class OnlineRoadModel:
    """
    Кусочно-плоская дорога по точкам контакта уверенных оценок трека (причинно, буфер ограничен).

    add(point) — точка контакта (центр модели − H/2 по нормали) на кадре с надёжным 3D;
    local_plane(xy) — плоскость z = a + b x + c y по точкам в радиусе radius от xy (взвешенно, ближние
    важнее), СКО — из невязок и числа точек; None, если точек меньше min_points.
    """

    def __init__(self, radius: float = 3000.0, min_points: int = 15, max_points: int = 2000,
                 sigma_floor: float = 20.0, prior: RoadPlane | None = None):
        self.radius = radius
        self.min_points = min_points
        self.sigma_floor = sigma_floor
        self.prior = prior
        self._points: deque = deque(maxlen=max_points)

    def __len__(self) -> int:
        return len(self._points)

    def add(self, point: np.ndarray) -> None:
        p = np.asarray(point, float)
        if np.all(np.isfinite(p)):
            self._points.append(p)

    def local_plane(self, xy: np.ndarray) -> RoadPlane | None:
        if len(self._points) >= self.min_points:
            pts = np.array(self._points)
            d = np.linalg.norm(pts[:, :2] - np.asarray(xy, float)[:2], axis=1)
            near = d <= self.radius
            if near.sum() >= self.min_points:
                p, dist = pts[near], d[near]
                w = 1.0 / (1.0 + (dist / (0.5 * self.radius)) ** 2)
                A = np.c_[np.ones(len(p)), p[:, 0] - xy[0], p[:, 1] - xy[1]]
                # уклон определяется, только если точки разнесены в плане; иначе — горизонтальная
                spread = np.linalg.svd((p[:, :2] - p[:, :2].mean(0)) * np.sqrt(w)[:, None], compute_uv=False)
                if spread.min() < 0.05 * self.radius * np.sqrt(w.sum()):
                    A = A[:, :1]
                coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], p[:, 2] * np.sqrt(w), rcond=None)
                resid = p[:, 2] - A @ coef
                # робастно: MAD невязок, плюс неопределённость среднего
                mad = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
                sigma = float(np.hypot(max(mad, self.sigma_floor), mad / np.sqrt(len(p))))
                # дальше от данных — неопределённее (экстраполяция)
                sigma *= 1.0 + float(dist.min()) / self.radius
                slope = coef[1:] if len(coef) == 3 else np.zeros(2)
                normal = np.array([-slope[0], -slope[1], 1.0])
                return RoadPlane(np.array([xy[0], xy[1], coef[0]]), normal, sigma)
        return self.prior
