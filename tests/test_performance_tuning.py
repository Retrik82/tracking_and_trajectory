"""Тесты новых компонентов производительности: бэкенды, центроид по полигону, автонастройка."""

import numpy as np
import pytest

from detection.yolo import _polygon_area_centroid, resolve_backend
from pipeline.autotune import apply_profile, check_signature, choose, hardware_signature
from pipeline.config import StreamConfig


def test_polygon_area_centroid_square():
    square = np.array([[10.0, 20.0], [110.0, 20.0], [110.0, 70.0], [10.0, 70.0]], dtype=np.float32)
    area, cu, cv = _polygon_area_centroid(square)
    assert area == pytest.approx(100 * 50)
    assert cu == pytest.approx(60.0)
    assert cv == pytest.approx(45.0)


def test_polygon_area_centroid_degenerate():
    line = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
    area, cu, cv = _polygon_area_centroid(line)
    assert np.isnan(area) and np.isnan(cu) and np.isnan(cv)


def test_resolve_backend_explicit_wins():
    assert resolve_backend("torch", "cpu") == "torch"
    assert resolve_backend("onnx", "cuda") == "onnx"


def test_resolve_backend_auto_gpu_is_torch():
    assert resolve_backend("auto", "cuda") == "torch"


def test_resolve_backend_auto_cpu_without_profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # нет configs/performance_profile.json — консервативно torch
    assert resolve_backend("auto", "cpu") == "torch"


def test_choose_respects_min_imgsz_and_budget_floor():
    measured = [
        {"backend": "openvino", "imgsz": 480, "ms_per_image": 60.0},
        {"backend": "openvino", "imgsz": 640, "ms_per_image": 110.0},
        {"backend": "torch", "imgsz": 960, "ms_per_image": 280.0},
    ]
    chosen = choose(measured, fps=30.0, min_imgsz=640)
    assert chosen["detector.backend"] == "openvino"
    assert chosen["detector.imgsz"] == 640            # 480 быстрее, но ниже порога качества
    # 1000/110 ≈ 9 изобр./с < 30 FPS → бюджет минимум 2 (две камеры для стерео)
    assert chosen["streaming.max_images_per_tick"] == 2


def test_choose_fast_hardware_no_budget():
    measured = [{"backend": "torch", "imgsz": 960, "ms_per_image": 5.0}]   # 200 изобр./с (GPU)
    chosen = choose(measured, fps=30.0)
    assert chosen["streaming.max_images_per_tick"] is None


def test_choose_errors_only_raises():
    with pytest.raises(RuntimeError):
        choose([{"backend": "onnx", "imgsz": 640, "error": "нет пакета"}], fps=30.0)


def test_apply_profile_sets_config_and_skips_private():
    cfg = StreamConfig()
    apply_profile(cfg, {"chosen": {"detector.backend": "onnx", "detector.imgsz": 640,
                                   "streaming.max_images_per_tick": 2, "_expected_ms_per_image": 100}})
    assert cfg.detector.backend == "onnx"
    assert cfg.detector.imgsz == 640
    assert cfg.streaming.max_images_per_tick == 2


def test_check_signature_detects_mismatch():
    profile = {"hardware": dict(hardware_signature(), cpu_model="другой CPU")}
    mismatches = check_signature(profile)
    assert any("cpu_model" in m for m in mismatches)
    assert check_signature({"hardware": hardware_signature()}) == []
