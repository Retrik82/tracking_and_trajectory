"""
Точка кабины без переобучения детектора: модель грузовика как 3D-коробки.

Грузовик — коробка L×W×H. Опорная точка — низ передней грани по центру (кабина).
Траектория опорной точки — B-сплайн P(t), ориентация коробки — касательная P'(t).
Контрольные точки и габариты подбираются least_squares так, чтобы рамка проекции
коробки совпала с рамками детектора на всех камерах. Рёбра рамки, упёршиеся
в край кадра, не используются: там видна только часть грузовика.

Почему это лучше центра маски: центр маски на каждом ракурсе — своя физическая
точка кузова, и 3D-точка прыгает при смене камер. Рамка коробки определяется
одной и той же геометрией на всех ракурсах. Кадры, где грузовик видит одна камера,
тоже участвуют: между участками с несколькими камерами они удерживают сплайн
(на синтетике — ~0.1 м). Траектория только по одной камере определяется плохо (1–5 м).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from reconstruction.bspline import BSplineTrajectory
from reconstruction.objective import (
    Cameras,
    ObjectiveData,
    ObservationGeometry,
    _support,
    build_trajectory,
    observation_cameras,
)
from reconstruction.regularization import regularization_residuals

# Углы коробки в осях грузовика: x — назад от кабины (0..L), y — вбок (±W/2), z — вверх (0..H)
_CORNERS = np.array([[x, y, z] for x in (0.0, 1.0) for y in (-0.5, 0.5) for z in (0.0, 1.0)])


@dataclass
class TruckDimensions:
    """Габариты грузовика в единицах мира."""

    length: float
    width: float
    height: float

    def as_array(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height], dtype=float)


# Седельный тягач с полуприцепом, в сантиметрах: начальное приближение.
DEFAULT_DIMENSIONS = TruckDimensions(length=1650.0, width=255.0, height=400.0)

# Узкие «физические» пределы (например 2.3–2.7 × 3–4.5 м) на этих данных хуже: рамки детектора
# систематически больше такой коробки, габариты упираются в границы, и медиана ошибки растёт
# с 3.3 до 3.8 м. Любые границы к тому же замедляют сходимость trf. Поэтому по умолчанию
# габариты свободны — это «эффективная коробка» для рамок детектора, а не реальный размер.
# Если реальные габариты известны из сцены Unreal, передайте их и fit_dimensions=False.
DEFAULT_DIMENSION_BOUNDS = None


# Центр кабины относительно опорной точки (низ передней грани по центру), единицы мира.
# Типичная кабина тягача: длина ~2.3 м, передняя грань от бампера до крыши ~0.3–3.8 м.
# Это допущение, а не измерение: реальные значения берутся из модели грузовика в Unreal.
CAB_CENTER_BACK = 120.0
CAB_CENTER_UP = 200.0


@dataclass
class BoxObservations:
    """
    Рамки детектора, по одной на (камера, кадр).

    edges — (M, 4): x1, x2, y1, y2 в пикселях; edge_valid — (M, 4): ребро не упирается в край кадра.
    """

    times: np.ndarray
    frames: np.ndarray
    camera_indices: np.ndarray
    edges: np.ndarray
    edge_valid: np.ndarray
    confidence: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.frames = np.asarray(self.frames, dtype=int)
        self.camera_indices = np.asarray(self.camera_indices, dtype=int)
        self.edges = np.asarray(self.edges, dtype=float).reshape(-1, 4)
        self.edge_valid = np.asarray(self.edge_valid, dtype=bool).reshape(-1, 4)
        self.confidence = np.asarray(self.confidence, dtype=float)
        count = len(self.times)
        for name in ("frames", "camera_indices", "confidence"):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} должен иметь форму (M,)")
        if self.edges.shape != (count, 4) or self.edge_valid.shape != (count, 4):
            raise ValueError("edges и edge_valid должны иметь форму (M, 4)")
        if count == 0:
            raise ValueError("Нет наблюдений")

    @classmethod
    def from_detections(cls, detections, image_sizes: dict[int, tuple[int, int]], margin: float = 3.0,
                        frame_offset: int = 0) -> "BoxObservations":
        """
        Из объектов Detection (detection.schema). Время — кадр видео, поза — кадр CSV = кадр + смещение.
        image_sizes — {камера: (ширина, высота)} фактического кадра её видео: ребро у края кадра своей
        камеры — не измерение (у камер разрешение может различаться).
        """
        detections = sorted(detections, key=lambda d: (d.frame, d.camera))
        missing = sorted({d.camera for d in detections} - set(image_sizes))
        if missing:
            raise ValueError(f"нет разрешения кадра для камер {missing} — края рамок не проверить")

        def valid(d):
            width, height = image_sizes[d.camera]
            return [d.x1 > margin, d.x2 < width - margin, d.y1 > margin, d.y2 < height - margin]

        return cls(
            times=[d.frame for d in detections],
            frames=[d.frame + frame_offset for d in detections],
            camera_indices=[d.camera for d in detections],
            edges=[[d.x1, d.x2, d.y1, d.y2] for d in detections],
            edge_valid=[valid(d) for d in detections],
            confidence=[d.confidence for d in detections],
        )

    def subset(self, mask: np.ndarray) -> "BoxObservations":
        return BoxObservations(self.times[mask], self.frames[mask], self.camera_indices[mask],
                               self.edges[mask], self.edge_valid[mask], self.confidence[mask])


def truck_axes(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Оси грузовика по скорости (N, 3): вперёд (с уклоном), вбок (горизонтально), вверх."""
    velocity = np.asarray(velocity, dtype=float)
    horizontal = velocity[:, :2]
    h_norm = np.maximum(np.linalg.norm(horizontal, axis=1, keepdims=True), 1e-9)
    forward = velocity / np.maximum(np.linalg.norm(velocity, axis=1, keepdims=True), 1e-9)
    side = np.c_[-horizontal[:, 1:2] / h_norm, horizontal[:, 0:1] / h_norm, np.zeros((len(velocity), 1))]
    up = np.cross(forward, side)
    up *= np.where(up[:, 2:3] < 0, -1.0, 1.0)
    return forward, side, up


def box_corners(cab_points: np.ndarray, velocity: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Углы коробки (N, 8, 3) для опорных точек кабины (N, 3)."""
    forward, side, up = truck_axes(velocity)
    length, width, height = dims
    return (cab_points[:, None, :]
            - _CORNERS[None, :, 0:1] * length * forward[:, None, :]
            + _CORNERS[None, :, 1:2] * width * side[:, None, :]
            + _CORNERS[None, :, 2:3] * height * up[:, None, :])


def project_box(geometry: ObservationGeometry, corners: np.ndarray, min_depth: float = 1.0) -> np.ndarray:
    """Рамка (M, 4) x1, x2, y1, y2 проекции углов (M, 8, 3); по одной камере на наблюдение."""
    us, vs = [], []
    for k in range(corners.shape[1]):
        pixels, _ = geometry.project(corners[:, k, :], min_depth=min_depth)
        us.append(pixels[:, 0])
        vs.append(pixels[:, 1])
    us, vs = np.stack(us, axis=1), np.stack(vs, axis=1)
    return np.c_[us.min(1), us.max(1), vs.min(1), vs.max(1)]


def initial_cab_trajectory(center: BSplineTrajectory, dims: TruckDimensions,
                           knot_spacing: float | None = None) -> BSplineTrajectory:
    """
    Начальное приближение: от центра масок сдвигаемся на L/2 вперёд и на H/2 вниз.
    Если задан knot_spacing, сплайн перестраивается на новой сетке узлов.
    """
    from reconstruction.bspline import fit_initial_bspline

    times = np.linspace(center.t_start, center.t_end, int(center.t_end - center.t_start) + 1)
    forward, _, up = truck_axes(center.derivative().evaluate(times))
    points = center.evaluate(times) + dims.length / 2 * forward - dims.height / 2 * up
    if knot_spacing is None:
        return fit_initial_bspline(times, points, num_control_points=len(center.control_points))
    return fit_initial_bspline(times, points, knot_spacing=knot_spacing)


def offset_point_trajectory(trajectory: BSplineTrajectory, back: float = CAB_CENTER_BACK,
                            up: float = CAB_CENTER_UP) -> BSplineTrajectory:
    """
    Траектория точки, жёстко связанной с грузовиком: опорная точка − back·вперёд + up·вверх.

    Оси грузовика берутся из касательной исходного сплайна. Результат — сплайн на том же
    узловом векторе, подогнанный по точкам на каждом кадре, поэтому производные
    (скорость, ускорение) считаются так же, как для исходной траектории.
    """
    times = np.linspace(trajectory.t_start, trajectory.t_end,
                        max(int(round(trajectory.t_end - trajectory.t_start)) + 1, 2 * len(trajectory.control_points)))
    forward, _, up_axis = truck_axes(trajectory.derivative().evaluate(times))
    points = trajectory.evaluate(times) - back * forward + up * up_axis
    from scipy.interpolate import BSpline

    basis = BSpline.design_matrix(times, trajectory.knots, trajectory.degree).toarray()
    control_points = np.linalg.lstsq(basis, points, rcond=None)[0]
    return build_trajectory(trajectory, control_points)


@dataclass
class CabFitResult:
    trajectory: BSplineTrajectory          # траектория точки кабины
    dimensions: TruckDimensions
    edge_errors_px: np.ndarray             # (M, 4) |рамка коробки − рамка детектора|, NaN для неиспользованных рёбер
    scipy_result: object


class CabObjective:
    """Остатки: рёбра рамок (с весом sqrt(confidence)) + регуляризация по ускорению и рывку."""

    def __init__(self, template: BSplineTrajectory, cameras: Cameras, observations: BoxObservations,
                 lambda_acceleration: float = 10.0, lambda_jerk: float = 100.0,
                 regularization_samples: int | None = None, min_depth: float = 1.0,
                 fixed_dimensions: np.ndarray | None = None):
        self.fixed_dimensions = None if fixed_dimensions is None else np.asarray(fixed_dimensions, dtype=float)
        inside = (observations.times >= template.t_start) & (observations.times <= template.t_end)
        self.observations = observations.subset(inside)
        self.template = template
        self.lambda_acceleration = lambda_acceleration
        self.lambda_jerk = lambda_jerk
        self.min_depth = min_depth
        obs = self.observations
        lookup = ObjectiveData(obs.times, obs.camera_indices, np.zeros((len(obs.times), 2)), obs.confidence, obs.frames)
        self.geometry = ObservationGeometry.from_cameras(observation_cameras(cameras, lookup))
        self.weights = np.sqrt(obs.confidence)[:, None] * obs.edge_valid
        samples = regularization_samples or max(2, int(template.t_end - template.t_start) // 2)
        self.regularization_times = np.linspace(template.t_start, template.t_end, samples)
        self.num_cp = len(template.control_points)

    @property
    def num_params(self) -> int:
        return 3 * self.num_cp + (0 if self.fixed_dimensions is not None else 3)

    def unpack(self, theta: np.ndarray) -> tuple[BSplineTrajectory, np.ndarray]:
        # Модуль: коробка с отрицательным габаритом — та же коробка (по ширине симметрична),
        # так не нужны ограничения, которые заметно замедляют сходимость trf
        dims = self.fixed_dimensions if self.fixed_dimensions is not None else np.abs(theta[3 * self.num_cp:])
        return build_trajectory(self.template, theta[:3 * self.num_cp].reshape(-1, 3)), dims

    def predicted_edges(self, theta: np.ndarray) -> np.ndarray:
        trajectory, dims = self.unpack(theta)
        t = self.observations.times
        corners = box_corners(trajectory.evaluate(t), trajectory.derivative().evaluate(t), dims)
        return project_box(self.geometry, corners, self.min_depth)

    def residuals(self, theta: np.ndarray) -> np.ndarray:
        trajectory, _ = self.unpack(theta)
        edges = (self.predicted_edges(theta) - self.observations.edges) * self.weights
        regularization = regularization_residuals(trajectory, self.regularization_times,
                                                  self.lambda_acceleration, self.lambda_jerk)
        return np.concatenate([edges.ravel(), regularization])

    def jac_sparsity(self) -> lil_matrix:
        count = len(self.observations.times)
        samples = len(self.regularization_times)
        sparsity = lil_matrix((4 * count + 6 * samples, self.num_params), dtype=np.int8)
        dims_columns = np.arange(3 * self.num_cp, self.num_params)
        for i, cps in enumerate(_support(self.template, self.observations.times)):
            columns = np.r_[(3 * cps[:, None] + np.arange(3)).ravel(), dims_columns]
            for row in range(4 * i, 4 * i + 4):
                sparsity[row, columns] = 1
        offset = 4 * count
        for _ in range(2):
            for s, cps in enumerate(_support(self.template, self.regularization_times)):
                for axis in range(3):
                    sparsity[offset + 3 * s + axis, 3 * cps + axis] = 1
            offset += 3 * samples
        return sparsity


def fit_cab_trajectory(
    initial: BSplineTrajectory,
    cameras: Cameras,
    observations: BoxObservations,
    dimensions: TruckDimensions = DEFAULT_DIMENSIONS,
    dimension_bounds: tuple[TruckDimensions, TruckDimensions] | None = DEFAULT_DIMENSION_BOUNDS,
    fit_dimensions: bool = True,
    lambda_acceleration: float = 10.0,
    lambda_jerk: float = 100.0,
    loss: str = "huber",
    f_scale: float = 5.0,
    max_nfev: int = 3000,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
) -> CabFitResult:
    """
    Подгоняет траекторию точки кабины и габариты к рамкам детектора.

    dimension_bounds — физические пределы габаритов (None — без ограничений).
    fit_dimensions=False — габариты фиксированы (например, известны из сцены Unreal).
    """
    dims0 = dimensions.as_array()
    objective = CabObjective(initial, cameras, observations, lambda_acceleration, lambda_jerk,
                             fixed_dimensions=None if fit_dimensions else dims0)
    num = 3 * objective.num_cp
    theta0 = initial.control_points.ravel().copy()
    lower = np.full(num, -np.inf)
    upper = np.full(num, np.inf)
    if fit_dimensions:
        if dimension_bounds is not None:
            dims_lower, dims_upper = dimension_bounds[0].as_array(), dimension_bounds[1].as_array()
        else:
            dims_lower, dims_upper = np.full(3, -np.inf), np.full(3, np.inf)
        theta0 = np.r_[theta0, np.clip(dims0, dims_lower + 1e-6, dims_upper - 1e-6)]
        lower = np.r_[lower, dims_lower]
        upper = np.r_[upper, dims_upper]
    bounded = np.any(np.isfinite(lower)) or np.any(np.isfinite(upper))

    result = least_squares(
        objective.residuals, theta0, jac="2-point", jac_sparsity=objective.jac_sparsity(),
        bounds=(lower, upper) if bounded else (-np.inf, np.inf), x_scale="jac", loss=loss, f_scale=f_scale, max_nfev=max_nfev, method="trf", ftol=ftol, xtol=xtol,
    )
    trajectory, dims = objective.unpack(result.x)
    errors = np.abs(objective.predicted_edges(result.x) - objective.observations.edges)
    errors[~objective.observations.edge_valid] = np.nan
    return CabFitResult(trajectory, TruckDimensions(*dims), errors, result)
