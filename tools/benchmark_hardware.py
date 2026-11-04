"""
Диагностика оборудования и замер скорости инференса YOLO (этапы 1 и 3 задания).

Запуск без аргументов НЕ требует весов и данных — только осмотр железа и окружения:
    python tools/benchmark_hardware.py
    python tools/benchmark_hardware.py --md docs/HARDWARE_PROFILE.md --json outputs/hardware.json

Замер YOLO (нужны веса; занимает минуты — экспорт ONNX при первом запуске):
    python tools/benchmark_hardware.py --bench-yolo --weights models/best.pt \
        --imgsz 960 640 480 --backends torch onnx

Результаты честные: каждый вариант реально прогоняется на этой машине.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- осмотр железа

def _try(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 — диагностика не должна падать из-за одного датчика
        return default


def cpu_info() -> dict:
    import psutil

    freq = _try(psutil.cpu_freq)
    info = {
        "model": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "max_freq_mhz": getattr(freq, "max", None),
        "current_freq_mhz": getattr(freq, "current", None),
        "load_percent": psutil.cpu_percent(interval=0.5),
    }
    # SIMD-возможности, которые реально использует PyTorch (NO_AVX / AVX2 / AVX512 ...)
    def torch_capability():
        import torch

        return torch.backends.cpu.get_cpu_capability()

    info["torch_cpu_capability"] = _try(torch_capability)
    if platform.system() == "Windows":
        name = _try(lambda: subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=20).stdout.strip())
        if name:
            info["model"] = name
    return info


def memory_info() -> dict:
    import psutil

    vm = psutil.virtual_memory()
    return {"total_gb": round(vm.total / 2**30, 2), "available_gb": round(vm.available / 2**30, 2),
            "percent_used": vm.percent}


def gpu_info() -> list[dict]:
    gpus = []
    if platform.system() == "Windows":
        out = _try(lambda: subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.AdapterRAM + '|' + $_.DriverVersion }"],
            capture_output=True, text=True, timeout=20).stdout)
        for line in (out or "").strip().splitlines():
            parts = line.split("|")
            if len(parts) == 3:
                vram = int(parts[1]) / 2**30 if parts[1].strip().isdigit() else None
                gpus.append({"name": parts[0].strip(), "adapter_ram_gb": round(vram, 2) if vram else None,
                             "driver": parts[2].strip()})
    smi = _try(lambda: subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20))
    if smi and smi.returncode == 0:
        for line in smi.stdout.strip().splitlines():
            gpus.append({"nvidia_smi": line.strip()})
    return gpus


def software_info() -> dict:
    sw = {"os": f"{platform.system()} {platform.release()} ({platform.version()})",
          "python": sys.version.split()[0]}

    def _ver(mod):
        m = __import__(mod)
        return getattr(m, "__version__", "?")

    for mod in ("numpy", "cv2", "ultralytics", "scipy"):
        sw[mod] = _try(lambda m=mod: _ver(m))

    def torch_block():
        import torch

        return {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "num_threads": torch.get_num_threads()}

    sw["torch"] = _try(torch_block)

    def ort_block():
        import onnxruntime as ort

        return {"version": ort.__version__, "providers": ort.get_available_providers()}

    sw["onnxruntime"] = _try(ort_block)

    def ov_block():
        import openvino as ov

        return {"version": ov.__version__, "devices": ov.Core().available_devices}

    sw["openvino"] = _try(ov_block)
    return sw


def quick_cpu_benchmark() -> dict:
    """Короткая оценка вычислительной мощности: GEMM float32 (цель SIMD/BLAS), ~1 с."""
    import numpy as np

    n = 1024
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    a @ b  # прогрев
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        a @ b
    dt = (time.perf_counter() - t0) / reps
    return {"gemm_1024_ms": round(dt * 1000, 1), "approx_gflops": round(2 * n**3 / dt / 1e9, 1)}


# ---------------------------------------------------------------- замер YOLO

def bench_yolo(weights: Path, imgsz_list: list[int], backends: list[str], device: str | None,
               iters: int, batch_sizes: list[int]) -> list[dict]:
    """Реальный инференс случайного кадра 1920×1080 на каждом (бэкенд, размер, batch)."""
    import numpy as np

    from detection.yolo import YoloDetector, YoloSettings

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    rows = []
    for backend in backends:
        for imgsz in imgsz_list:
            try:
                det = YoloDetector(YoloSettings(weights_path=weights, imgsz=imgsz, device=device, backend=backend))
            except Exception as e:  # noqa: BLE001 — бэкенд недоступен: фиксируем и идём дальше
                rows.append({"backend": backend, "imgsz": imgsz, "error": str(e)[:200]})
                continue
            for batch in batch_sizes:
                images = [frame] * batch
                for _ in range(2):  # прогрев
                    det.detect_images(images)
                t0 = time.perf_counter()
                for _ in range(iters):
                    det.detect_images(images)
                dt = (time.perf_counter() - t0) / iters
                rows.append({"backend": det.backend, "imgsz": imgsz, "batch": batch,
                             "ms_per_call": round(dt * 1000, 1), "ms_per_image": round(dt * 1000 / batch, 1),
                             "images_per_s": round(batch / dt, 2)})
                print(f"  {det.backend:8s} imgsz={imgsz} batch={batch}: "
                      f"{dt * 1000 / batch:7.1f} мс/изобр., {batch / dt:5.2f} изобр./с", flush=True)
            del det
    return rows


# ---------------------------------------------------------------- отчёт

def to_markdown(data: dict) -> str:
    cpu, mem, sw = data["cpu"], data["memory"], data["software"]
    lines = ["# Профиль оборудования", "",
             f"Снято: {data['timestamp']} — автоматически, `python tools/benchmark_hardware.py`.", "",
             "## CPU", "",
             f"- Модель: {cpu['model']}",
             f"- Ядра: {cpu['physical_cores']} физических / {cpu['logical_cpus']} логических",
             f"- Частота: максимум {cpu['max_freq_mhz']} МГц",
             f"- SIMD (возможности PyTorch): {cpu['torch_cpu_capability']}",
             f"- Загрузка в момент замера: {cpu['load_percent']} %", "",
             "## Память", "",
             f"- Всего {mem['total_gb']} ГБ, свободно {mem['available_gb']} ГБ ({mem['percent_used']} % занято)", "",
             "## GPU", ""]
    if data["gpus"]:
        for g in data["gpus"]:
            lines.append(f"- {json.dumps(g, ensure_ascii=False)}")
    else:
        lines.append("- не обнаружены")
    lines += ["", "## Программная среда", ""]
    for k, v in sw.items():
        lines.append(f"- {k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v}")
    if data.get("cpu_benchmark"):
        b = data["cpu_benchmark"]
        lines += ["", "## Быстрый замер CPU", "",
                  f"- GEMM 1024×1024 float32: {b['gemm_1024_ms']} мс ≈ {b['approx_gflops']} GFLOPS"]
    if data.get("yolo_benchmark"):
        lines += ["", "## Инференс YOLO (реальные замеры на этой машине)", "",
                  "| бэкенд | imgsz | batch | мс/изобр. | изобр./с |", "|---|---|---|---|---|"]
        for r in data["yolo_benchmark"]:
            if "error" in r:
                lines.append(f"| {r['backend']} | {r['imgsz']} | — | ошибка: {r['error'][:80]} | — |")
            else:
                lines.append(f"| {r['backend']} | {r['imgsz']} | {r['batch']} | {r['ms_per_image']} | {r['images_per_s']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=None, help="сохранить JSON")
    p.add_argument("--md", type=Path, default=None, help="сохранить Markdown (например docs/HARDWARE_PROFILE.md)")
    p.add_argument("--bench-yolo", action="store_true", help="замер инференса (нужны веса)")
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    p.add_argument("--imgsz", type=int, nargs="+", default=[960, 640, 480])
    p.add_argument("--backends", nargs="+", default=["torch", "onnx"])
    p.add_argument("--batch", type=int, nargs="+", default=[1, 2])
    p.add_argument("--device", default=None)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args(argv)

    data = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cpu": cpu_info(), "memory": memory_info(), "gpus": gpu_info(), "software": software_info(),
            "cpu_benchmark": _try(quick_cpu_benchmark)}
    if args.bench_yolo:
        if not args.weights.is_file():
            sys.exit(f"нет весов {args.weights} — замер YOLO пропущен (положите best.pt в models/)")
        print("Замер YOLO (каждая строка — реальный прогон):", flush=True)
        data["yolo_benchmark"] = bench_yolo(args.weights, args.imgsz, args.backends, args.device,
                                            args.iters, args.batch)
    text = to_markdown(data)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
