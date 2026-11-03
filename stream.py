"""
Пеленг: потоковая система мультикамерного 3D-трекинга — единая точка входа.

    python stream.py replay     --data tmp/project2 --video        # записанные видео в реальном темпе, YOLO, видео с наложением
    python stream.py live       --config configs/live.json        # живые потоки (cameras.live_sources)
    python stream.py detections --data tmp/project2 --detections outputs/main/detections
                                                                   # готовые детекции без YOLO (алгоритмы, метрики)
    python stream.py variants   --data tmp/project2 --detections outputs/main/detections
                                                                   # сравнение архитектур A–E на одних детекциях
    python stream.py scenarios                                     # обязательные сценарии ТЗ на синтетике
    python stream.py plots      --track outputs/stream/stream_track.csv --data tmp/project2
    python stream.py config                                        # напечатать конфигурацию по умолчанию (JSON)

Общие параметры режимов replay/live/detections: --config file.json, --set раздел.ключ=значение, --out.
Подробно: README.md (раздел «Потоковая система»), docs/ARCHITECTURE.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd in ("replay", "live", "detections"):
        from pipeline.streaming import main as stream_main

        stream_main(["--mode", cmd, *rest])
    elif cmd == "variants":
        from pipeline.variants import main as variants_main

        variants_main(rest)
    elif cmd == "scenarios":
        from tools.run_scenarios import main as scenarios_main

        scenarios_main(rest)
    elif cmd == "plots":
        from pipeline.stream_plots import main as plots_main

        plots_main(rest)
    elif cmd == "config":
        from pipeline.config import StreamConfig

        print(StreamConfig.default_json())
    else:
        raise SystemExit(f"неизвестная команда {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
