"""
Пеленг: 3D-траектория грузовика по синхронным видео пяти камер.

Основной результат — траектория центра кабины от трекера реального времени
(3D-модель грузовика по рамкам YOLO, только прошлые и текущий кадры). Офлайн-варианты
(центр маски, оптимизация по пикселям, модель коробки по всему видео) — для сравнения.
Подробно — README.md.

Полный прогон:
    .venv/bin/python main.py --weights models/best.pt --data tmp/project2 --video

Отдельные шаги (детекции и промежуточные результаты переиспользуются с диска):
    .venv/bin/python main.py --steps cab,compare
    .venv/bin/python main.py --steps compare --video

Шаги: detect, center, optimize, cab, realtime, compare (+ video). Подробнее: --help.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.run import build_main_parser, run  # noqa: E402


def main(argv=None) -> None:
    run(build_main_parser().parse_args(argv))


if __name__ == "__main__":
    main()
