"""
Форма грузовика из 3D-модели (FBX) для сопровождения.

Модель CargoTruck.fbx: Y вверх, кабина на +Z, X вбок, сантиметры. Габариты в файле
6.69 × 1.68 × 2.42 м; это жёсткий грузовик (кабина и кузов на одной раме), без сцепки.

Параметры сцены (откалиброваны по рамкам YOLO при положении и курсе из CSV цели,
медиана невязки рёбер 7.5 px; их стоит сверить с актором в Unreal):
- масштаб актора ×3.186 — в сцене грузовик 21.3 × 5.35 × 7.72 м;
- точка CSV цели в модели: +345 см вдоль (передний край кабины, кабина до +336)
  и +93 см вверх от начала координат модели (в единицах модели, до масштаба).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from reconstruction.fbx_reader import load_meshes

SCENE_SCALE = 3.186
CSV_POINT_MODEL = (345.0, 93.0)          # (вдоль, вверх) в единицах модели FBX, до масштаба
CAB_MESH = "Truck"            # тягач с рамой; кабина — его часть впереди кузова
CARGO_MESH = "Cargo"


@dataclass
class TruckModel:
    """
    Точки формы в осях грузовика (вперёд, вбок, вверх) относительно центра габаритного
    параллелепипеда модели, в единицах мира сцены (с учётом масштаба).
    """

    hull: np.ndarray                     # (K, 3) вершины выпуклой оболочки: рамка проекции определяется ими
    dimensions: np.ndarray               # (3,) длина, ширина, высота
    csv_point: np.ndarray                # (3,) точка, которую пишет CSV цели
    cab_center: np.ndarray               # (3,) центр габаритов кабины

    @classmethod
    def from_fbx(cls, path: Path, scale: float = SCENE_SCALE,
                 csv_point_model: tuple[float, float] = CSV_POINT_MODEL, cab_mesh: str = CAB_MESH,
                 cargo_mesh: str = CARGO_MESH) -> "TruckModel":
        scene = load_meshes(path)
        to_truck = lambda v: np.c_[v[:, 2], v[:, 0], v[:, 1]]     # noqa: E731  (Z, X, Y) -> (вперёд, вбок, вверх)
        vertices = to_truck(np.vstack([m.vertices for m in scene.meshes])) * scale
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        center = (lo + hi) / 2
        hull = vertices[ConvexHull(vertices).vertices] - center
        cab = [m for m in scene.meshes if m.name == cab_mesh]
        if not cab:
            raise ValueError(f"В модели нет части {cab_mesh!r}: {[m.name for m in scene.meshes]}")
        cab_v = to_truck(cab[0].vertices) * scale
        cargo = [m for m in scene.meshes if m.name == cargo_mesh]
        if cargo:
            # Часть «Truck» включает раму под кузовом; кабина — то, что впереди кузова
            cargo_front = (to_truck(cargo[0].vertices) * scale)[:, 0].max()
            cab_v = cab_v[cab_v[:, 0] > cargo_front]
        cab_center = (cab_v.min(axis=0) + cab_v.max(axis=0)) / 2 - center
        csv_point = np.array([csv_point_model[0], 0.0, csv_point_model[1]]) * scale - center
        return cls(hull, hi - lo, csv_point, cab_center)

    def to_world(self, center: np.ndarray, forward: np.ndarray, side: np.ndarray, up: np.ndarray,
                 local: np.ndarray) -> np.ndarray:
        """Точки local (K, 3) в осях грузовика -> мир при центре center и осях (3,)."""
        return center + local[:, 0:1] * forward + local[:, 1:2] * side + local[:, 2:3] * up
