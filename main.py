"""
Пеленг: 3D-траектория грузовика по синхронным видео пяти (или N) камер — итоговая точка входа.

Запуск без аргументов выполняет все нужные процессы по порядку:

  1. detect    YOLO по каждому видео -> детекции (кешируются на диске)            ТЗ п.1
  2. center    центр маски -> робастная триангуляция -> начальный сплайн (офлайн)  ТЗ п.2
  3. optimize  сплайн по пикселям всех камер (офлайн, для сравнения)               ТЗ п.3
  4. cab       коробка грузовика по рамкам всего видео (офлайн, для сравнения)     ТЗ п.3
  5. realtime  ОСНОВНОЙ: фильтр Калмана с 3D-моделью, только прошлые кадры;
               закрытые рёбра, пропуски, пределы ускорения                        ТЗ п.3, п.4
  6. approx    базовый вариант: обычная аппроксимация 3D-точек полиномом
               (причинный по окну — реальное время; глобальный — офлайн)           сравнение
  7. compare   метрики против CSV цели: report.md, metrics.json, metrics_summary.csv,
               графики (ошибка, координаты от времени, погрешность), сводка в консоль  ТЗ п.5, п.6
  8. video     контрольное видео с рамкой/силуэтом модели, точкой кабины, следом   ТЗ п.5

Примеры:
    .venv/bin/python main.py                                   # всё, включая видео
    .venv/bin/python main.py --no-video                        # всё без видео
    .venv/bin/python main.py --steps realtime,approx,compare   # только трекер, база и отчёт (офлайн-шаги с диска)
    .venv/bin/python main.py --redetect                        # пересчитать детекции (новые веса)
    .venv/bin/python main.py --cameras 1,2,3                   # произвольный набор камер
    .venv/bin/python main.py --data tmp/synthetic --out outputs/synthetic --steps center,optimize,cab,realtime,approx,compare
                                                               # синтетика (tools/make_synthetic_data.py), без видео и YOLO

Данные: --data (видео *.FinalImage.CineCameraActor{N}.mp4 — разрешение берётся из каждого, CSV камер и цели),
модели: --weights, --truck-fbx.
Все параметры: --help. Потоковая система реального времени (replay / live / detections, сравнение A–E, сценарии) — stream.py.
Подробно о потоке обработки и файлах — README.md, обзор кода — docs/PIPELINE_REVIEW.md, архитектура — docs/ARCHITECTURE.md.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.run import ALL_STEPS, build_main_parser, run  # noqa: E402


def main(argv=None) -> int:
    parser = build_main_parser()
    parser.description = __doc__
    args = parser.parse_args(argv)

    missing = [str(p) for p in (args.data,) if not Path(p).exists()]
    if missing:
        parser.error(f"нет папки с данными: {', '.join(missing)} (укажите --data)")
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if "detect" in steps and not Path(args.weights).exists():
        if steps == ALL_STEPS and (Path(args.out) / "detections" / "cam1.csv").exists():
            print(f"весов {args.weights} нет — шаг detect пропущен, детекции берутся из {args.out / 'detections'}")
            args.steps = ",".join(s for s in steps if s != "detect")
        else:
            parser.error(f"нет весов YOLO {args.weights} (укажите --weights или уберите шаг detect)")

    started = time.time()
    run(args)
    print(f"готово за {time.time() - started:.0f} с: результаты в {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
