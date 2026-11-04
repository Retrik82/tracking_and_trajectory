# Профиль оборудования

Снято: 2026-11-04 17:07:06 — автоматически, `python tools/benchmark_hardware.py`.

## CPU

- Модель: AMD Ryzen 5 4500U with Radeon Graphics
- Ядра: 6 физических / 6 логических
- Частота: максимум 2375.0 МГц
- SIMD (возможности PyTorch): AVX2
- Загрузка в момент замера: 35.9 %

## Память

- Всего 7.37 ГБ, свободно 1.6 ГБ (78.3 % занято)

## GPU

- {"name": "AMD Radeon(TM) Graphics", "adapter_ram_gb": 0.5, "driver": "31.0.21924.61"}

## Программная среда

- os: Windows 11 (10.0.26300)
- python: 3.14.4
- numpy: 2.5.4
- cv2: 5.0.0
- ultralytics: 8.4.200
- scipy: 1.18.1
- torch: {"version": "2.15.0+cpu", "cuda_available": false, "cuda_version": null, "num_threads": 6}
- onnxruntime: {"version": "1.31.1", "providers": ["AzureExecutionProvider", "CPUExecutionProvider"]}
- openvino: {"version": "2026.4.1-22982-e213a147257-releases/2026/4", "devices": ["CPU", "GPU"]}

## Быстрый замер CPU

- GEMM 1024×1024 float32: 10.3 мс ≈ 207.7 GFLOPS

## Инференс YOLO (реальные замеры на этой машине)

| бэкенд | imgsz | batch | мс/изобр. | изобр./с |
|---|---|---|---|---|
| torch | 960 | 1 | 282.0 | 3.55 |
| torch | 960 | 2 | 349.3 | 2.86 |
| torch | 640 | 1 | 453.6 | 2.2 |
| torch | 640 | 2 | 376.5 | 2.66 |
| torch | 480 | 1 | 133.0 | 7.52 |
| torch | 480 | 2 | 106.3 | 9.4 |
| onnx | 960 | 1 | 483.0 | 2.07 |
| onnx | 960 | 2 | 231.8 | 4.31 |
| onnx | 640 | 1 | 137.2 | 7.29 |
| onnx | 640 | 2 | 162.0 | 6.17 |
| onnx | 480 | 1 | 122.7 | 8.15 |
| onnx | 480 | 2 | 105.3 | 9.5 |
| openvino | 960 | 1 | 249.9 | 4.0 |
| openvino | 960 | 2 | 282.5 | 3.54 |
| openvino | 640 | 1 | 137.0 | 7.3 |
| openvino | 640 | 2 | 134.3 | 7.45 |
| openvino | 480 | 1 | 68.4 | 14.62 |
| openvino | 480 | 2 | 64.1 | 15.6 |
