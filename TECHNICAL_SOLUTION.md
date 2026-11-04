# Техническое решение: параметры камер, разрешение видео, экспорт координат

## Что изменено

1. **Параметры камер из CSV** (`reconstruction/camera.py: read_camera_csv`). Колонки ищутся по именам:
   `Loc_X/Y/Z` — положение (единицы Unreal, см), `Rot_Roll/Pitch/Yaw` — углы (градусы → радианы),
   фокус в мм — колонка `Focal_Length_mm` в каждой строке (сцена 2, может меняться по кадрам) или пара
   `Focal length;35` в заголовке (сцена 1). Некорректные строки (не числа, пусто, неполная строка, фокус ≤ 0,
   повтор кадра) пропускаются и перечисляются в предупреждении; нет колонки/фокуса — ошибка с текстом заголовка.
   Файл камеры N — `object_transform_CineCameraActor{N}.csv`.
2. **Матрица постоянная 23.76 × 13.365 мм** (`SENSOR_WIDTH_MM`, `SENSOR_HEIGHT_MM`). Пересчёт мм → пиксели —
   только `pixel_focal_lengths`: `fx = f·W/23.76`, `fy = f·H/13.365`, `cx = W/2`, `cy = H/2`; K собирается
   в `Camera.from_physical`. Кадр с соотношением сторон не как у матрицы (обрезка/поля) — ошибка.
3. **Разрешение — отдельно для каждого видео** (`reconstruction/data_io.py: probe_video_resolution`):
   метаданные OpenCV проверяются первым декодированным кадром (при расхождении верен кадр). Видео камеры N —
   `*.FinalImage.CineCameraActor{N}.mp4` (любая сцена; два подходящих файла — ошибка). Без видео — `image_size`
   из `detections/meta.json` (пишет шаг detect). Не определить — ошибка, подстановки нет. В потоке размер
   каждого кадра сверяется с камерой; смена разрешения пересчитывает K.
4. **Убраны фиксированные размеры**: `IMAGE_WIDTH/HEIGHT`, `cameras.width/height/sensor_*`, `image_width/height`
   в настройках трекера и планировщика — края кадра, видимость, рёбра рамок, запись видео берут размер у камеры.
5. **Экспорт координат** `trajectory.csv` (`pipeline/export.py`) — в `stream.py` (рядом с `stream_track.csv`)
   и в `main.py` (шаг realtime). Алгоритм трекинга не менялся: экспорт только читает готовые оценки.

## trajectory.csv

| колонка | смысл |
|---|---|
| frame_id | кадр видео |
| timestamp | время, с (кадр / FPS; live — часы захвата) |
| object_id | 1 (сопровождается один грузовик) |
| x, y, z | центр кабины — основная точка трекера реального времени, **метры**, мировая система Unreal Engine (левая, Z вверх) |
| camera_id | камеры с детекцией, вошедшие в оценку (через пробел); пусто — прогноз без измерений |
| status | measured / partial / predicted (кадры инициализации без координат не пишутся) |
| position_std_m | СКО положения по ковариации фильтра, м |

## Как проверить

```
py -3.14 -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python main.py --data <папка с видео и CSV> --out outputs\main --cameras 6,9 --steps detect --imgsz 480
.venv\Scripts\python stream.py detections --data <папка> --detections outputs\main\detections --set cameras.ids=[6,9] --no-pacing --out outputs\stream
```

Результат: `outputs\stream\trajectory.csv`, `stream_track.csv`, `performance_report.md`. Веса `models/best.pt`
и видео в архив не входят (размер); FBX грузовика нет — трекер использует коробку 16.5 × 2.55 × 4 м.
Тесты: `tests/test_camera_params.py` (CSV, разрешение, K, интеграция, экспорт).

## Проверено фактически

- pytest: 206 passed, 8 skipped (пропущены тесты данных сцены 1 — их на машине нет).
- Синтетика 1920×1080: результаты офлайн-шагов и трекера реального времени побайтно совпадают с master.
- Реальные видео сцены 2 (YOLO, камеры 6 и 9): `docs/results/scene2/` — trajectory.csv (810 кадров с
  координатами), отчёт производительности, кадры с наложением. Эталона (CSV цели) в сцене 2 нет —
  точность в метрах не проверялась. Подробно — FULL_TECHNICAL_ANALYSIS.md.
