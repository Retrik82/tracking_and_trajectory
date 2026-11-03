# Веса моделей

Файлы весов (`*.pt`, `*.onnx`) в git не коммитятся — они в `.gitignore`.

Текущая модель: YOLO26s-seg, один класс `truck`, imgsz 960, дообучена на кадрах
видео проекта (поэтому проверка на этих же видео — оценка на обучающих данных).

Подключение:
1. Положите файл весов сюда как `models/best.pt` (ultralytics принимает только расширение `.pt`).
2. Запустите проверку:
   ```
   .venv/bin/python scripts/yolo_eval.py --weights models/best.pt --data tmp/project2 --video
   ```
