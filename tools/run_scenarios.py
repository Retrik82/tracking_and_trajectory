"""
Обязательные сценарии ТЗ (п.6) на синтетике: генерация данных и сравнение вариантов архитектуры.

Для каждого сценария (tools/make_synthetic_data.py: SCENARIOS) создаются данные и детекции,
затем в режиме detections прогоняются варианты (по умолчанию A, C, D, E — pipeline/variants.py),
и собирается сводная таблица: ошибка положения (медиана / p95 / макс), MAE по осям, доля кадров с
оценкой, потери трека, время повторного захвата, среднее число активных камер.

Все числа — на синтетических данных (коробка вместо грузовика, детекции без YOLO), они показывают
поведение алгоритмов в сценариях, а не точность на реальном видео.

    python tools/run_scenarios.py --root tmp/scenarios --out outputs/scenarios
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import StreamConfig                    # noqa: E402
from pipeline.variants import VARIANTS, run_variants          # noqa: E402
from tools.make_synthetic_data import SCENARIOS, generate    # noqa: E402

log = logging.getLogger("peleng.scenarios")

SCENARIO_TITLES = {
    "cams5": "все 5 камер видят объект", "cams4": "только 4 камеры", "cams3": "только 3 камеры",
    "cams2": "только 2 камеры", "cam1": "только 1 камера (по умолчанию: трек не стартует без второй камеры)",
    "cam1_mono": "только 1 камера, старт по высоте рамки (tracker.mono_init = true, quality = mono)", "occlusion": "частичное перекрытие (съёживание рамки, пропуски камер)",
    "blackout": "полное исчезновение на 30 и 60 кадров", "reappear": "повторные появления после коротких потерь",
    "sharp_turn": "резкий поворот (петля 90 × 40 м)", "accel_brake": "разгон и торможение (8 с период)",
    "speed_change": "изменение скорости, остановка 3 с", "false_detection": "ошибочная детекция одной камеры (8 %)",
    "outlier": "выброс в координатах центроида (5 %)", "desync": "рассинхронизация камер 2 и 4 на 1–2 кадра",
    "handover": "последовательный переход между зонами видимости (одна камера 150–330)", "base": "базовая сцена",
}


def _f(x, d=2):
    return "—" if x is None or x != x else f"{x:.{d}f}"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("tmp/scenarios"))
    p.add_argument("--out", type=Path, default=Path("outputs/scenarios"))
    p.add_argument("--scenarios", default=",".join(k for k in SCENARIOS if k != "base"))
    p.add_argument("--variants", default="A,C,D,E")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    base = StreamConfig.load(args.config, args.set)
    keys = args.variants.split(",")
    table = {}
    for name in args.scenarios.split(","):
        sc = SCENARIOS[name]
        data_dir, out_dir = args.root / name, args.out / name
        info = generate(data_dir, out_dir, sc)
        cfg = base.copy()
        cfg.cameras.ids = list(sc.cameras)
        for dotted, value in sc.overrides.items():
            section, key = dotted.split(".")
            setattr(getattr(cfg, section), key, value)
        print(f"сценарий {name}: {SCENARIO_TITLES.get(name, '')}; камеры {sc.cameras}", flush=True)
        results = run_variants(cfg, data_dir, out_dir / "detections", out_dir / "variants", keys,
                               source_delays=sc.source_delays or None)
        table[name] = {k: {kk: vv for kk, vv in m.items() if kk != "error_series"} for k, m in results.items()}
        table[name]["_info"] = info
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scenarios.json").write_text(json.dumps(table, ensure_ascii=False, indent=2, default=float))
    lines = ["# Сценарии ТЗ на синтетике", "",
             "> Синтетические данные (коробка, детекции без YOLO). Числа характеризуют поведение алгоритмов в "
             "сценариях, не точность на реальном видео. Варианты: " +
             "; ".join(f"{k} — {VARIANTS[k][0]}" for k in keys) + ".", ""]
    for name, res in table.items():
        lines += [f"## {name}: {SCENARIO_TITLES.get(name, '')}", "",
                  "| вариант | кадров | доля с оценкой | медиана, м | p95 | макс | MAE X / Y / Z | скорость, медиана м/с "
                  "| потерь | predicted | повт. захват, кадров | активных камер | детекций/тик |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in keys:
            m = res.get(k)
            if not m:
                continue
            e, ax, pf = m["error_3d_m"], m["per_axis_m"], m["performance"]
            lines.append(f"| {k} | {m['frames_evaluated']} | {m['availability'] * 100:.0f} % | {_f(e.get('median'))} | "
                         f"{_f(e.get('p95'))} | {_f(e.get('max'))} | {_f(ax['x'].get('mae'))} / {_f(ax['y'].get('mae'))} / "
                         f"{_f(ax['z'].get('mae'))} | {_f(m['speed_error_mps'].get('median'))} | {m['track_losses']} | "
                         f"{m['predicted_frames']} | {_f(m['reacquisition_frames']['mean'], 1)} | "
                         f"{_f(pf['mean_active_cameras'])} | {_f(pf['detections_per_tick'])} |")
        lines.append("")
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print((args.out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
