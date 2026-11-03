"""Формат одной 2D-детекции объекта на кадре."""

from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class Detection:
    camera: int
    frame: int                          # номер кадра видео (до применения смещения к CSV)
    confidence: float
    x1: float                           # рамка, пиксели
    y1: float
    x2: float
    y2: float
    point_u: float                      # точка объекта, которая идёт в триангуляцию
    point_v: float
    point_source: str                   # mask_centroid | bbox_center | keypoint
    mask_area_px: float = np.nan
    mask_centroid_u: float = np.nan
    mask_centroid_v: float = np.nan
    keypoint_u: float = np.nan          # на будущее: YOLO-pose
    keypoint_v: float = np.nan
    num_candidates: int = 1             # сколько объектов класса нашла модель на кадре
    edge_clipped: bool = False          # рамка касается края кадра
    too_small: bool = False             # маска меньше порога
    unreliable: bool = False            # не использовать в триангуляции

    @classmethod
    def columns(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "Detection":
        types = {f.name: f.type for f in fields(cls)}
        out = {}
        for name, value in row.items():
            t = types[name]
            if t in (bool, "bool"):
                out[name] = str(value) in ("True", "true", "1")
            elif t in (int, "int"):
                out[name] = int(value)
            elif t in (str, "str"):
                out[name] = value
            else:
                out[name] = float(value)
        return cls(**out)
