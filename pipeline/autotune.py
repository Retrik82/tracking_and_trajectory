"""
Автоматическая настройка производительности под конкретный компьютер (Auto Performance Tuning).

`stream.py ... --auto-tune`   короткая калибровка: осмотр железа, замер доступных бэкендов
                              и размеров изображения на реальных весах, выбор конфигурации,
                              сохранение профиля в configs/performance_profile.json.
`stream.py ... --auto-config` применить сохранённый профиль; перед применением проверяется,
                              что оборудование то же (модель CPU, число ядер, наличие CUDA).

Профиль применяется ПЕРЕД переопределениями командной строки (--set), поэтому ручные
настройки пользователя всегда сильнее автоматических.

Выбор варианта: среди замеренных берётся самый быстрый с imgsz ≥ min_imgsz (порог качества;
меньший размер ухудшает детекцию мелких объектов — см. docs/PERFORMANCE_REPORT.md).
Бюджет изображений на тик считается из фактической скорости: capacity = 1000/ms_per_image,
max_images_per_tick = max(1, floor(capacity / fps)) — очередь инференса не растёт бесконтрольно.
"""

import json
import logging
import platform
import time
from pathlib import Path

log = logging.getLogger("peleng.autotune")

PROFILE_PATH = Path("configs/performance_profile.json")


def hardware_signature() -> dict:
    import psutil

    sig = {"platform": platform.system(), "cpu_model": platform.processor(),
           "logical_cpus": psutil.cpu_count(logical=True)}
    try:
        import torch

        sig["torch"] = torch.__version__
        sig["cuda"] = torch.cuda.is_available()
    except ImportError:
        sig["torch"], sig["cuda"] = None, False
    return sig


def _available_backends() -> list[str]:
    out = ["torch"]
    try:
        import onnxruntime  # noqa: F401

        out.append("onnx")
    except ImportError:
        pass
    try:
        import openvino  # noqa: F401

        out.append("openvino")
    except ImportError:
        pass
    return out


def measure_variants(weights: Path, imgsz_list: list[int], backends: list[str], device: str | None,
                     iters: int = 3) -> list[dict]:
    """Короткий честный замер каждого (бэкенд, imgsz) на кадре 1920×1080. Ошибки фиксируются."""
    import numpy as np

    from detection.yolo import YoloDetector, YoloSettings

    frame = np.random.default_rng(0).integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    rows = []
    for backend in backends:
        for imgsz in imgsz_list:
            try:
                det = YoloDetector(YoloSettings(weights_path=weights, imgsz=imgsz, device=device,
                                                backend=backend, retina_masks=False))
                det.detect_images([frame])                       # прогрев
                t0 = time.perf_counter()
                for _ in range(iters):
                    det.detect_images([frame])
                ms = (time.perf_counter() - t0) / iters * 1000
                rows.append({"backend": det.backend, "imgsz": imgsz, "ms_per_image": round(ms, 1)})
                log.info("auto-tune: %s imgsz=%d — %.1f мс/изобр.", det.backend, imgsz, ms)
                del det
            except Exception as e:  # noqa: BLE001 — бэкенд недоступен: не вариант, но и не авария
                rows.append({"backend": backend, "imgsz": imgsz, "error": str(e)[:200]})
                log.warning("auto-tune: %s imgsz=%d недоступен: %s", backend, imgsz, e)
    return rows


def choose(measured: list[dict], fps: float, min_imgsz: int = 640) -> dict:
    """
    Самый быстрый вариант с imgsz ≥ min_imgsz. Бюджет изображений на тик — из скорости,
    но не меньше 2: для инициализации и стерео-измерений фильтру нужны две камеры в одном
    тике (бюджет 1 оставил бы трекер в INITIALIZING навсегда). Быстрое железо — без бюджета.
    """
    ok = [r for r in measured if "ms_per_image" in r and r["imgsz"] >= min_imgsz]
    if not ok:
        ok = [r for r in measured if "ms_per_image" in r]
    if not ok:
        raise RuntimeError("auto-tune: ни один вариант инференса не работает")
    best = min(ok, key=lambda r: r["ms_per_image"])
    capacity = 1000.0 / best["ms_per_image"]
    budget = max(2, int(capacity / fps)) if capacity < 5 * fps else None
    return {"detector.backend": best["backend"], "detector.imgsz": best["imgsz"],
            "detector.retina_masks": False,
            "streaming.max_images_per_tick": budget,
            "_expected_ms_per_image": best["ms_per_image"],
            "_expected_images_per_s": round(capacity, 1)}


def run_auto_tune(cfg, profile_path: Path = PROFILE_PATH, min_imgsz: int = 640) -> dict:
    """Полная калибровка; сохраняет и возвращает профиль. Требует весов детектора."""
    weights = Path(cfg.detector.weights)
    sizes = sorted({cfg.detector.imgsz, 960, 640, 480})
    measured = measure_variants(weights, sizes, _available_backends(), cfg.detector.device)
    chosen = choose(measured, cfg.cameras.fps, min_imgsz)
    profile = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "hardware": hardware_signature(),
               "measured": measured, "chosen": chosen}
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("auto-tune: выбран %s imgsz=%s (~%.1f мс/изобр.), профиль сохранён в %s",
             chosen["detector.backend"], chosen["detector.imgsz"], chosen["_expected_ms_per_image"], profile_path)
    return profile


def load_profile(profile_path: Path = PROFILE_PATH) -> dict:
    if not profile_path.is_file():
        raise FileNotFoundError(f"нет профиля {profile_path} — сначала запустите с --auto-tune")
    return json.loads(profile_path.read_text(encoding="utf-8"))


def check_signature(profile: dict) -> list[str]:
    """Расхождения текущего железа с профилем (пустой список — совместимо)."""
    current = hardware_signature()
    saved = profile.get("hardware", {})
    return [f"{k}: профиль {saved.get(k)!r}, сейчас {current.get(k)!r}"
            for k in ("platform", "cpu_model", "logical_cpus", "cuda") if saved.get(k) != current.get(k)]


def apply_profile(cfg, profile: dict) -> None:
    """Перенести выбранные параметры в конфигурацию (до переопределений --set, ключи «раздел.ключ»)."""
    for key, value in profile.get("chosen", {}).items():
        if key.startswith("_"):
            continue
        section, _, name = key.partition(".")
        setattr(getattr(cfg, section), name, value)
    log.info("профиль производительности применён: %s",
             {k: v for k, v in profile.get("chosen", {}).items() if not k.startswith("_")})
