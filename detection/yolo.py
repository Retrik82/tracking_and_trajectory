"""Детектор на YOLO (ultralytics). ultralytics импортируется только при создании детектора."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detection.schema import Detection


class WeightsNotFoundError(RuntimeError):
    pass


@dataclass
class YoloSettings:
    weights_path: Path
    imgsz: int = 960
    conf: float = 0.25
    device: str | None = None           # None — выбрать автоматически (mps / cuda / cpu)
    target_class: str = "truck"
    min_mask_area_px: float = 400.0
    edge_margin_px: float = 2.0
    min_reliable_conf: float = 0.5


@dataclass
class FrameResult:
    detection: Detection | None
    polygon: np.ndarray | None          # (K, 2) контур маски выбранного объекта, пиксели


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloDetector:
    def __init__(self, settings: YoloSettings):
        weights = Path(settings.weights_path)
        if not weights.is_file():
            raise WeightsNotFoundError(
                f"Файл весов не найден: {weights}. Положите best.pt в models/ "
                f"и передайте путь через --weights."
            )
        if weights.suffix != ".pt":
            raise WeightsNotFoundError(
                f"ultralytics принимает только файлы .pt, а передан {weights.name}. "
                f"Переименуйте файл, например в models/best.pt."
            )

        from ultralytics import YOLO

        self.settings = settings
        self.device = settings.device or _pick_device()
        self.model = YOLO(str(weights))
        names = self.model.names
        matches = [i for i, n in names.items() if n == settings.target_class]
        if not matches:
            raise ValueError(f"В модели нет класса {settings.target_class!r}, есть: {names}")
        self.class_id = matches[0]
        self.is_segmentation = self.model.task == "segment"

    def detect_video(self, video_path: Path, camera: int) -> Iterator[FrameResult]:
        """Детекции по всем кадрам видео; на каждый кадр — объект с максимальной уверенностью."""
        s = self.settings
        stream = self.model.predict(
            source=str(video_path), stream=True, imgsz=s.imgsz, conf=s.conf,
            classes=[self.class_id], device=self.device, retina_masks=True, verbose=False,
        )
        for frame, result in enumerate(stream):
            yield self._convert(result, camera, frame)

    def _convert(self, result, camera: int, frame: int) -> FrameResult:
        s = self.settings
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return FrameResult(None, None)

        conf = boxes.conf.cpu().numpy()
        best = int(np.argmax(conf))
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(float)
        height, width = result.orig_shape

        area = cu = cv = np.nan
        polygon = None
        if self.is_segmentation and result.masks is not None:
            mask = result.masks.data[best].cpu().numpy() > 0.5
            ys, xs = np.nonzero(mask)
            area = float(len(xs))
            if len(xs):
                cu, cv = float(xs.mean()) + 0.5, float(ys.mean()) + 0.5
            polygon = np.asarray(result.masks.xy[best], dtype=np.float32)

        if np.isfinite(cu):
            point_u, point_v, source = cu, cv, "mask_centroid"
        else:
            point_u, point_v, source = (x1 + x2) / 2, (y1 + y2) / 2, "bbox_center"

        m = s.edge_margin_px
        edge_clipped = bool(x1 <= m or y1 <= m or x2 >= width - m or y2 >= height - m)
        too_small = bool(np.isfinite(area) and area < s.min_mask_area_px)
        unreliable = edge_clipped or too_small or float(conf[best]) < s.min_reliable_conf

        detection = Detection(
            camera=camera, frame=frame, confidence=float(conf[best]),
            x1=x1, y1=y1, x2=x2, y2=y2,
            point_u=point_u, point_v=point_v, point_source=source,
            mask_area_px=area, mask_centroid_u=cu, mask_centroid_v=cv,
            num_candidates=len(conf),
            edge_clipped=edge_clipped, too_small=too_small, unreliable=unreliable,
        )
        return FrameResult(detection, polygon)
