"""
Детектор на YOLO (ultralytics). ultralytics импортируется только при создании детектора.

Бэкенды инференса (settings.backend):
  torch      исходные веса .pt (PyTorch);
  onnx       ONNX Runtime CPU — на CPU заметно быстрее torch; .onnx экспортируется из .pt
             один раз и кэшируется рядом с весами (dynamic shapes: один файл на все imgsz);
  openvino   OpenVINO (если установлен) — экспорт и кэш аналогично;
  auto       cuda → torch; иначе onnx, если установлен onnxruntime, иначе torch.

Постобработка масок (settings.retina_masks):
  True       маски полного разрешения (как раньше): точный центроид, но дорого на CPU;
  False      центроид и площадь считаются по полигону контура (masks.xy) — в разы дешевле;
             полигон ultralytics всегда отдаёт в координатах исходного кадра.
"""

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
    device: str | None = None           # None — выбрать автоматически (cuda / mps / cpu)
    backend: str = "auto"               # auto | torch | onnx | openvino
    half: bool = False                  # FP16 (только GPU)
    retina_masks: bool = True           # маски полного разрешения (False — центроид по полигону)
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


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _openvino_available() -> bool:
    try:
        import openvino  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_backend(backend: str, device: str) -> str:
    """
    auto: на GPU — torch; на CPU — бэкенд из сохранённого профиля производительности
    (configs/performance_profile.json, создаётся `stream.py --auto-tune`), если он есть
    и установлен; иначе — torch (консервативно, без сюрпризов).
    """
    if backend != "auto":
        return backend
    if device in ("cuda", "mps"):
        return "torch"
    try:
        import json

        profile = json.loads(Path("configs/performance_profile.json").read_text(encoding="utf-8"))
        candidate = profile.get("chosen", {}).get("detector.backend", "torch")
    except (OSError, ValueError):
        return "torch"
    if candidate == "onnx" and _onnxruntime_available():
        return "onnx"
    if candidate == "openvino" and _openvino_available():
        return "openvino"
    return "torch"


def export_model(weights: Path, backend: str, imgsz: int) -> Path:
    """Экспорт .pt в формат бэкенда с кэшем рядом с весами. Возвращает путь модели для YOLO()."""
    from ultralytics import YOLO

    if backend == "onnx":
        target = weights.with_suffix(".onnx")
        if not target.is_file() or target.stat().st_mtime < weights.stat().st_mtime:
            # dynamic=True: один файл обслуживает любые imgsz и batch; simplify=False —
            # без onnxslim (не тянуть зависимости при закрытой сети)
            YOLO(str(weights)).export(format="onnx", dynamic=True, imgsz=imgsz, simplify=False, verbose=False)
        return target
    if backend == "openvino":
        target = weights.parent / f"{weights.stem}_openvino_model"
        if not target.is_dir():
            YOLO(str(weights)).export(format="openvino", dynamic=True, imgsz=imgsz, verbose=False)
        return target
    raise ValueError(f"неизвестный бэкенд {backend!r}")


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
        self.backend = resolve_backend(settings.backend, self.device)
        if self.backend == "onnx" and not _onnxruntime_available():
            raise RuntimeError("backend=onnx: пакет onnxruntime не установлен (pip install onnxruntime)")
        if self.backend == "openvino" and not _openvino_available():
            raise RuntimeError("backend=openvino: пакет openvino не установлен (pip install openvino)")
        model_path = weights if self.backend == "torch" else export_model(weights, self.backend, settings.imgsz)
        self.model = YOLO(str(model_path), task="segment")
        names = self.model.names
        matches = [i for i, n in names.items() if n == settings.target_class]
        if not matches and len(names) == 1:
            # однаклассовая модель с другим именем класса (в разметке best.pt-4 — «track»): это и есть цель
            import logging

            logging.getLogger("peleng").warning("в модели нет класса %r — единственный класс %r принят за цель",
                                                settings.target_class, next(iter(names.values())))
            matches = list(names)
        if not matches:
            raise ValueError(f"В модели нет класса {settings.target_class!r}, есть: {names}")
        self.class_id = matches[0]
        self.is_segmentation = self.model.task == "segment"

    def _predict_kwargs(self) -> dict:
        s = self.settings
        kw = dict(imgsz=s.imgsz, conf=s.conf, classes=[self.class_id], device=self.device,
                  retina_masks=s.retina_masks, verbose=False)
        if s.half and self.device == "cuda":
            kw["half"] = True
        return kw

    def detect_video(self, video_path: Path, camera: int) -> Iterator[FrameResult]:
        """Детекции по всем кадрам видео; на каждый кадр — объект с максимальной уверенностью."""
        stream = self.model.predict(source=str(video_path), stream=True, **self._predict_kwargs())
        for frame, result in enumerate(stream):
            yield self._convert(result, camera, frame)

    def detect_images(self, images: list[np.ndarray], cameras: list[int] | None = None,
                      frames: list[int] | None = None) -> list[FrameResult]:
        """Пакетный инференс списка кадров BGR (потоковый конвейер и замеры)."""
        if not images:
            return []
        cameras = cameras or [0] * len(images)
        frames = frames or [0] * len(images)
        results = self.model.predict(source=images, **self._predict_kwargs())
        return [self._convert(res, cam, fr) for res, cam, fr in zip(results, cameras, frames)]

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
            polygon = np.asarray(result.masks.xy[best], dtype=np.float32)
            if s.retina_masks:
                mask = result.masks.data[best].cpu().numpy() > 0.5
                ys, xs = np.nonzero(mask)
                area = float(len(xs))
                if len(xs):
                    cu, cv = float(xs.mean()) + 0.5, float(ys.mean()) + 0.5
            elif len(polygon) >= 3:
                area, cu, cv = _polygon_area_centroid(polygon)

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


def _polygon_area_centroid(polygon: np.ndarray) -> tuple[float, float, float]:
    """Площадь (px²) и центроид полигона (формула шнурования); polygon (K, 2)."""
    x, y = polygon[:, 0], polygon[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area2 = cross.sum()
    if abs(area2) < 1e-6:
        return float("nan"), float("nan"), float("nan")
    cu = float(((x + xn) * cross).sum() / (3.0 * area2))
    cv = float(((y + yn) * cross).sum() / (3.0 * area2))
    return float(abs(area2) / 2.0), cu, cv
