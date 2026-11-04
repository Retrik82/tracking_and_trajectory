# Развёртывание и запуск

## Установка

```bash
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

- Веса детектора: положить `best.pt` в `models/` (в git не хранится).
- 3D-модель: `models/CargoTruck.fbx` (без неё — коробка 16.5 × 2.55 × 4 м, точность хуже).
- ffmpeg нужен только для записи видео с наложением: `winget install Gyan.FFmpeg`.
- `onnxruntime` и `openvino` — ускорение CPU-инференса; без них работает torch.

## Диагностика компьютера (без весов и данных)

```bash
python tools/benchmark_hardware.py --md docs/HARDWARE_PROFILE.md
```

С замером YOLO (нужны веса, несколько минут):

```bash
python tools/benchmark_hardware.py --bench-yolo --imgsz 960 640 480 --backends torch onnx openvino
```

## Первый запуск: калибровка под компьютер

```bash
python stream.py replay --data D:\project2 --auto-tune
```

Замеряет доступные бэкенды и размеры, выбирает конфигурацию, сохраняет
`configs/performance_profile.json` и сразу запускает прогон с ней.

## Эксплуатация (live)

```bash
python stream.py live --config configs/live.json --auto-config
```

`configs/live.json` — источники RTSP и параметры (пример в репозитории).
`--auto-config` применяет сохранённый профиль производительности с проверкой,
что оборудование не поменялось.

Ручная настройка всегда сильнее профиля:

```bash
python stream.py live --config configs/live.json --auto-config --set detector.imgsz=960 --set detector.backend=torch
```

## Проверка качества (replay без потерь)

```bash
python stream.py replay --data D:\project2 --no-pacing --auto-config --set triangulation.mode=none
```

`triangulation.mode=none` — вариант A, лучший на реальных данных (см. FINAL_REPORT, раздел 11).

## Что смотреть после прогона

- `outputs/<папка>/performance_report.md` — тики/с, изображения YOLO/с, задержка p50/p95,
  вытесненные кадры, состояния планировщика камер.
- `outputs/<папка>/stream_track.csv` — REALTIME и SMOOTHED траектории.
- `stream.py variants --data ... --detections ...` — сравнение архитектур A–E с метриками
  точности против CSV цели.

## Известные ограничения (ноябрь 2026, машина разработки)

- CPU AMD Ryzen 5 4500U, без CUDA: лучший замер — OpenVINO, 640 px, ~9 изобр./с.
  150 изобр./с (5 камер × 30 FPS) на этом железе недостижимы; система работает адаптивно
  (бюджет изображений на тик + планировщик камер), трекер даёт оценку на каждом тике.
- ОЗУ 7.4 ГБ: не запускать параллельно несколько прогонов с YOLO.
- Качество на меньших imgsz проверяйте по `stream.py variants` на реальных детекциях,
  а не только по FPS (см. docs/PERFORMANCE_REPORT.md).
