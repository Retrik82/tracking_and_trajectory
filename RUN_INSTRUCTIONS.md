# Как запустить (Windows, без GPU)

## 1. Что куда положить

```
tracking_and_trajectory/
├── data/                                    ← видео и CSV сцены (в git не попадают)
│   ├── Way2.FinalImage.CineCameraActor1.mp4 … CineCameraActor9.mp4
│   ├── object_transform_CineCameraActor1.csv … CineCameraActor9.csv
│   └── matrix_size.txt                      (необязательно; матрица 23.76 × 13.365 мм зашита в код)
└── models/
    └── best.pt                              ← веса YOLO (файл best.pt-4 из Telegram переименовать в best.pt)
```

- Имя видео: любой префикс сцены, главное — окончание `.FinalImage.CineCameraActor{N}.mp4`; номер N
  связывает видео с `object_transform_CineCameraActor{N}.csv`. В папке не должно быть двух видео одной камеры.
- Разрешение видео указывать не нужно — берётся из каждого файла. Фокус, позиция и углы — из CSV.
- `models/CargoTruck.fbx` необязателен (без него трекер использует коробку 16.5 × 2.55 × 4 м).
- ultralytics принимает только расширение `.pt` — поэтому `best.pt-4` → `best.pt`.

## 2. Установка (один раз)

Нужен Python 3.12+ (проверено на 3.14):

```
winget install Python.Python.3.14 --scope user
cd tracking_and_trajectory
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install numpy scipy opencv-python matplotlib psutil pytest
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install ultralytics openvino
```

(или `pip install -r requirements.txt`). Проверка: `.venv\Scripts\python -m pytest -q` (~8–10 мин).
Во всех командах ниже в PowerShell сначала `$env:PYTHONUTF8=1` (иначе русские логи ломают консоль).

## 3. Детекция YOLO по видео (долго: ~12 мин на камеру на CPU)

```
.venv\Scripts\python main.py --data data --out outputs\scene2_all --steps detect --cameras 1,2,3,4,5,6,7,8,9 --imgsz 480 --backend openvino
```

Результат: `outputs\scene2_all\detections\cam{N}.csv`, контуры масок, `meta.json` (там же разрешение
каждого видео). Повторный запуск пропускает готовые камеры (`--redetect` — пересчитать).
Быстрее — две копии параллельно на разные камеры с разными `--out`, потом сложить `cam*` в одну папку
(как в `outputs/run_demo9.sh`).

## 4. Трекер реального времени и координаты в CSV

Конфигурация камер (файл `outputs\scene2_all\demo_cfg.json`):

```
{"cameras": {"ids": [1, 2, 3, 4, 5, 6, 7, 8, 9]}, "triangulation": {"mode": "none"}}
```

```
.venv\Scripts\python stream.py detections --data data --detections outputs\scene2_all\detections --config outputs\scene2_all\demo_cfg.json --no-pacing --out outputs\scene2_all\stream
```

Результат в `outputs\scene2_all\stream\`:
- `trajectory.csv` — `frame_id, timestamp, object_id, x, y, z, camera_id, status, position_std_m`;
  x, y, z — центр кабины, **метры**, мировая система Unreal (левая, Z вверх);
- `stream_track.csv` — полный трек (скорость, курс, сглаженный сплайн, камеры), `performance_report.md`.

Живой режим без готовых детекций (YOLO по кадрам на лету; на CPU успевает 2–3 камеры):
`stream.py replay --data data --set cameras.ids=[6,9] --set detector.imgsz=480 --set detector.backend=openvino`.

## 5. Посмотреть глазами (окно в реальном темпе)

```
.venv\Scripts\python tools\demo_live.py --data data --detections outputs\scene2_all\detections --config outputs\scene2_all\demo_cfg.json
```

Сетка камер (до 9), синяя проекция модели по оценке трекера, вид сверху, X/Y/Z. Трекер считается вживую,
детекции — готовые. Записать в файл: `--save outputs\demo.mp4 --no-window`. Пробел — пауза, Esc/q — выход.

## 6. Проверка геометрии (без эталона)

```
.venv\Scripts\python tools\check_scene_geometry.py . data outputs\scene2_all\detections outputs\scene2_all\stream\stream_track.csv outputs\check 6,9 1300,1600
```

Ошибка перепроецирования стерео-пары и кадры с наложением в `outputs\check`.

## Заметки

- В сцене 2 нет CSV цели (`object_transform_SM_AA_track.csv`) — ошибку в метрах посчитать нельзя;
  `main.py` без `--steps detect` (шаги center/compare) требует этот файл.
- В весах best.pt-4 класс называется `track` — детектор берёт единственный класс модели автоматически.
- Подробности: `TECHNICAL_SOLUTION.md`, `FULL_TECHNICAL_ANALYSIS.md`.
