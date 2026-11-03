"""
Пеленг: 3D-траектория грузовика по синхронным видео пяти камер.

Основной результат — траектория точки кабины (модель 3D-коробки по рамкам YOLO).
Для сравнения считаются варианты по центру маски: триангуляция, начальный сплайн
и оптимизация по пикселям.

Полный прогон:
    .venv/bin/python main.py --weights models/best.pt --data tmp/project2 --video

Отдельные шаги (детекции и промежуточные результаты переиспользуются с диска):
    .venv/bin/python main.py --steps cab,compare
    .venv/bin/python main.py --steps compare --video

Шаги: detect, center, optimize, cab, compare (+ video). Подробнее: --help.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.cab_pipeline import build_main_parser, run  # noqa: E402


def main(argv=None) -> None:
    run(build_main_parser().parse_args(argv))


if __name__ == "__main__":
    main()
