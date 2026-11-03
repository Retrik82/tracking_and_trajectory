# Результаты прогонов

Файлы в `synthetic/`, `scenarios/` и `demo/` получены на синтетической сцене (`tools/make_synthetic_data.py`): коробка вместо
грузовика, детекции без YOLO. Они показывают работу кода и поведение алгоритмов в сценариях,
**а не точность на реальном видео**. Как получить то же на реальных данных — `docs/FINAL_REPORT.md`, раздел 10.

| папка | что | команда |
|---|---|---|
| `synthetic/variants/` | сравнение архитектур A–E: `report.md`, `variants.json`, графики | `python stream.py variants --data tmp/synthetic --detections outputs/synthetic/detections --out outputs/synthetic/variants` |
| `synthetic/stream/` | потоковый прогон (планировщик + триангуляция + Калман + сплайн): `stream_track.csv` (REALTIME и SMOOTHED TRACK по тикам), `smoothed_track.csv`, `performance_report.md/json`, `config_used.json`, `plots/` (координаты, скорость/ускорение, ошибки и неопределённость, камеры и задержка) | `python stream.py detections --data tmp/synthetic --detections outputs/synthetic/detections --out outputs/synthetic/stream --no-pacing` |
| `scenarios/` | обязательные сценарии ТЗ по вариантам A, C, D, E: `report.md`, `scenarios.json` | `python stream.py scenarios` |
| `synthetic/compare/` | офлайн-конвейер `main.py` (центр маски, оптимизация, коробка, трекер, аппроксимация): `report.md`, `metrics.json`, `metrics_summary.csv`, графики 01–09 | `python main.py --data tmp/synthetic --out outputs/synthetic --no-video` |

Столбцы `stream_track.csv`: `frame, t_s, status, quality, x y z` (центр модели, см), `ref_*` (точка CSV),
`cab_*` (центр кабины), `vx vy vz speed_mps accel_mps2 heading_deg position_std_m`, `cams_detected cams_used
point_cams point_nis reproj_px active_cams detected_cams latency_ms`, `smoothed_frame sx sy sz svx svy svz
s_accel_mps2` (SMOOTHED TRACK на своём кадре), `excluded` (камеры вне допуска синхронизации), `rejected_cams`
(рамки, отброшенные гейтом).

## Реальные данные (`real/`, 4 ноября 2026)

Прогоны на реальных видео `D:\project2` с `best.pt` и `CargoTruck.fbx` (YOLO обучена на этих же видео);
описание — `docs/FINAL_REPORT.md`, раздел 11.

| папка | что | команда |
|---|---|---|
| `real/main/` | офлайн-конвейер и трекер: `report.md`, `metrics.json`, `metrics_summary.csv`, графики `compare/01–09`, отчёт детектора `center/report.md`; `metrics_summary_box_no_fbx.csv` — тот же трекер с коробкой вместо FBX | `python main.py --data D:\project2`; коробка: `--steps realtime,compare --truck-fbx models\NO_SUCH.fbx --out outputs\main_box` |
| `real/variants/` | варианты A–E на реальных детекциях: `report.md`, `variants.json`, графики | `python stream.py variants --data D:\project2 --detections outputs\main\detections --out outputs\main\variants` |
| `real/stream/` | потоковый конвейер с умолчаниями на реальных детекциях: `performance_report.md/json`, `config_used.json`, `plots/` | `python stream.py detections --data D:\project2 --detections outputs\main\detections --no-pacing --out outputs\stream_detections` |
| `real/diag/` | диагностика (`edges_heading`, `E_no_point`, `C_no_heading`): отчёты производительности и конфигурации | `python stream.py detections ... --set triangulation.mode=none` и т. д. (FINAL_REPORT 11.3) |
| `real/replay/` | replay с YOLO на CPU: в реальном темпе и `--no-pacing`; кадры видео | `python stream.py replay --data D:\project2 --video [--no-pacing]` |
