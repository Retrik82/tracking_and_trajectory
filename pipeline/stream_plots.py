"""
Графики потокового прогона (ТЗ 7): X(t), Y(t), Z(t), V(t), ускорение, ошибка по осям, неопределённость,
число активных камер. Вход — stream_track.csv и (для ошибок) CSV цели.

    python -m pipeline.stream_plots --track outputs/stream/stream_track.csv --data tmp/project2 --out outputs/stream/plots
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from pipeline.center import AQUA, BLUE, GRID, INK_2, MAGENTA, MUTED, ORANGE, YELLOW, _style, find_gaps
from pipeline.variants import ground_truth, reference_from_center


def read_track(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)

    def col(name, dtype=float):
        if dtype is float:
            return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows])
        return [r[name] for r in rows]

    return {
        "frame": col("frame", float).astype(int), "t": col("t_s"), "status": np.array(col("status", str)),
        "center": np.c_[col("x"), col("y"), col("z")], "ref": np.c_[col("ref_x"), col("ref_y"), col("ref_z")],
        "v": np.c_[col("vx"), col("vy"), col("vz")], "speed": col("speed_mps"), "accel": col("accel_mps2"),
        "heading": col("heading_deg"), "std": col("position_std_m"),
        "active": np.array([len(r["active_cams"].split()) for r in rows]),
        "detected": np.array([len(r["detected_cams"].split()) for r in rows]),
        "point_cams": np.array([len(r["point_cams"].split()) for r in rows]),
        "latency": col("latency_ms"), "s_frame": col("smoothed_frame"),
        "s_pos": np.c_[col("sx"), col("sy"), col("sz")], "s_v": np.c_[col("svx"), col("svy"), col("svz")],
        "s_accel": col("s_accel_mps2"),
    }


def plot_stream(track: dict, out: Path, data_dir: Path | None, fps: float = 30.0, upm: float = 100.0,
                frame_offset: int = 0, model=None, dims=None) -> list[Path]:
    plt = _style()
    out.mkdir(parents=True, exist_ok=True)
    fr, t = track["frame"], track["t"]
    n = int(fr.max()) + 1
    truth = None
    if data_dir is not None and (Path(data_dir) / "object_transform_SM_AA_track.csv").exists():
        truth = ground_truth(Path(data_dir), n, fps, upm, frame_offset)
    ref = track["ref"] / upm
    # сглаженная точка CSV на её собственном кадре
    s_ref = np.full((n, 3), np.nan)
    s_speed = np.full(n, np.nan)
    s_acc = np.full(n, np.nan)
    heading_by_frame = {int(f): h for f, h in zip(fr, track["heading"])}
    if dims is None:
        dims = np.array([1650.0, 255.0, 400.0])
    for i in range(len(fr)):
        sf = track["s_frame"][i]
        if np.isfinite(sf) and np.all(np.isfinite(track["s_pos"][i])):
            k = int(round(sf))
            if 0 <= k < n:
                s_ref[k] = reference_from_center(track["s_pos"][i], heading_by_frame.get(k, track["heading"][i]),
                                                 model, dims) / upm
                s_speed[k] = np.linalg.norm(track["s_v"][i])
                s_acc[k] = track["s_accel"][i]
    s_t = np.arange(n) / fps
    files = []

    # 1. координаты от времени
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for i, ax_name in enumerate("XYZ"):
        ax = axes[i]
        if truth is not None:
            ax.plot(truth["frames"] / fps, truth["gt"][:, i] / upm, color=INK_2, lw=2.2, label="цель (CSV, только оценка)")
        ax.plot(t, ref[:, i], color=BLUE, lw=1.6, label="REALTIME TRACK (Калман)")
        ax.plot(s_t, s_ref[:, i], color=AQUA, lw=1.4, ls="--", label="SMOOTHED TRACK (B-сплайн, с задержкой)")
        pred = track["status"] == "predicted"
        for a, b in find_gaps(~pred):
            ax.axvspan(t[a], t[b], color=ORANGE, alpha=0.15, lw=0)
        ax.set_ylabel(f"{ax_name}, м")
    axes[0].set_title("Координаты от времени (оранжевым — прогноз без измерений)")
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)
    axes[-1].set_xlabel("Время, с")
    fig.tight_layout()
    files.append(out / "01_coordinates.png")
    fig.savefig(files[-1])
    plt.close(fig)

    # 2. скорость и ускорение
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    if truth is not None:
        a1.plot(truth["frames"] / fps, truth["speed"], color=INK_2, lw=2, label="цель")
        a2.plot(truth["frames"] / fps, truth["accel"], color=INK_2, lw=2, label="цель")
    a1.plot(t, track["speed"], color=BLUE, lw=1.5, label="Калман")
    a1.plot(s_t, s_speed, color=AQUA, lw=1.3, ls="--", label="сплайн")
    a2.plot(t, track["accel"], color=BLUE, lw=1.3, label="Калман")
    a2.plot(s_t, s_acc, color=AQUA, lw=1.3, ls="--", label="сплайн")
    a1.set_ylabel("V, м/с")
    a2.set_ylabel("|a|, м/с²")
    a2.set_xlabel("Время, с")
    a1.set_title("Скорость и ускорение: фильтр (без задержки) и сплайн (с задержкой)")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    a2.set_ylim(0, min(np.nanpercentile(np.r_[track["accel"], s_acc], 99) * 1.5 if np.any(np.isfinite(track["accel"])) else 10, 40))
    fig.tight_layout()
    files.append(out / "02_speed_accel.png")
    fig.savefig(files[-1])
    plt.close(fig)

    # 3. ошибка по осям и неопределённость
    if truth is not None:
        gt = truth["gt"] / upm
        err_rt = ref - gt[fr]
        err_s = s_ref - gt
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        for i, ax_name in enumerate("XYZ"):
            axes[i].plot(t, err_rt[:, i], color=BLUE, lw=1.3, label="Калман")
            axes[i].plot(s_t, err_s[:, i], color=AQUA, lw=1.1, ls="--", label="сплайн")
            axes[i].axhline(0, color=MUTED, lw=1)
            axes[i].set_ylabel(f"Δ{ax_name}, м")
        e3 = np.linalg.norm(err_rt, axis=1)
        horiz = np.linalg.norm(err_rt[:, :2], axis=1)
        axes[3].fill_between(t, 0, 2 * track["std"], color=BLUE, alpha=0.15, lw=0, label="2·СКО фильтра (горизонталь)")
        axes[3].plot(t, horiz, color=YELLOW, lw=1.3, label="ошибка по горизонтали")
        axes[3].plot(t, e3, color=MAGENTA, lw=1.0, label="ошибка 3D")
        axes[3].set_ylabel("м")
        axes[3].set_xlabel("Время, с")
        axes[0].set_title("Ошибка по осям относительно CSV цели и заявленная неопределённость")
        axes[0].legend(fontsize=8)
        axes[3].legend(fontsize=8)
        fig.tight_layout()
        files.append(out / "03_errors_uncertainty.png")
        fig.savefig(files[-1])
        plt.close(fig)

    # 4. камеры, статус, задержка
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a1.step(t, track["active"], where="mid", color=BLUE, lw=1.5, label="активных камер (план)")
    a1.step(t, track["detected"], where="mid", color=ORANGE, lw=1.2, label="камер с детекцией")
    a1.step(t, track["point_cams"], where="mid", color=AQUA, lw=1.0, label="камер в триангуляции")
    a1.set_yticks(range(0, int(max(track["active"].max(), 1)) + 2))
    a1.set_ylabel("Камер")
    a1.legend(fontsize=8, loc="upper right")
    a1.set_title("Камеры по тикам и задержка кадр → оценка")
    a2.plot(t, track["latency"], color=INK_2, lw=1)
    a2.set_ylabel("Задержка, мс")
    a2.set_xlabel("Время, с")
    for code, color in (("predicted", ORANGE), ("partial", MAGENTA)):
        for a, b in find_gaps(track["status"] != code):
            a1.axvspan(t[a], t[b], color=color, alpha=0.15, lw=0)
    fig.tight_layout()
    files.append(out / "04_cameras_latency.png")
    fig.savefig(files[-1])
    plt.close(fig)
    return files


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", type=Path, required=True)
    p.add_argument("--data", type=Path, default=None, help="папка с CSV цели (для ошибок)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args(argv)
    files = plot_stream(read_track(args.track), args.out or args.track.parent / "plots", args.data, args.fps)
    print("\n".join(str(f) for f in files))


if __name__ == "__main__":
    main()
