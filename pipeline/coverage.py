"""
Вид сверху: поля зрения камер, где грузовик загорожен, и оценка препятствий.

Геометрии сцены (деревьев, гор) нет, поэтому препятствия оцениваются по данным —
«выскабливанием пространства» (space carving) на сетке вида сверху:
- луч камера -> грузовик на кадре, где камера грузовик нашла, проходит через свободное
  место: клетки вдоль него помечаются свободными;
- луч на кадре, где грузовик в поле зрения, но не найден, где-то упирается в препятствие:
  клетки вдоль него, не помеченные свободными, — зона, где может быть препятствие.
  Одна камера не говорит, на каком расстоянии по лучу оно стоит, поэтому зона широкая;
- клетка, в которую упираются «загороженные» лучи двух разных камер, — препятствие,
  подтверждённое двумя камерами (пересечение зон);
- тень сектора: от подтверждённого препятствия, а если на луче его нет — за концом зоны
  (дальше препятствия свет точно не проходит; внутри зоны — неизвестно).

Положение грузовика — из CSV цели (только для оценки). Оценка двумерная: высоту препятствий
и камер не учитывает (луч с высокой камеры может пройти над низким препятствием).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.center import BLUE, CAMERA_COLORS, CAMERAS, INK_2, NUM_VIDEO_FRAMES, ORANGE, _style
from reconstruction.data_io import make_camera

SEEN, SEEN_FLAGGED, OCCLUDED, OUT_OF_VIEW = 0, 1, 2, 3
COVERAGE_LABELS = {
    SEEN: "в поле зрения, найден",
    SEEN_FLAGGED: "найден с флагом (край кадра, мала, ненадёжна)",
    OCCLUDED: "в поле зрения, но не найден — загорожен",
    OUT_OF_VIEW: "вне поля зрения",
}
COVERAGE_COLORS = {SEEN: BLUE, SEEN_FLAGGED: "#86b6ef", OCCLUDED: ORANGE, OUT_OF_VIEW: "#d6d5d0"}
OBSTACLE_COLOR = "#3d3c39"

CELL_M = 2.0               # размер клетки сетки, м
NEAR_CAMERA_M = 5.0        # у самой камеры препятствие не ищем
NEAR_TRUCK_M = 12.0        # конец луча — сам грузовик (21 м длиной, точка CSV у передней грани)
ANGLE_BIN_DEG = 0.5        # шаг по углу при рисовании света и тени сектора


def _image_size(args, cam: int):
    """Разрешение видео камеры (определено в pipeline/run.py); None — make_camera определит по видео."""
    return (getattr(args, "image_sizes", None) or {}).get(cam)


def camera_coverage(args, detections, gt: np.ndarray) -> dict[int, np.ndarray]:
    """
    Состояние каждой камеры на каждом кадре: найден / найден с флагом / загорожен / вне кадра.

    «В поле зрения» — точка цели проецируется в кадр перед камерой (поза камеры с этого кадра).
    """
    state = {}
    for cam in CAMERAS:
        s = np.full(NUM_VIDEO_FRAMES, OUT_OF_VIEW)
        for f in range(NUM_VIDEO_FRAMES):
            d = detections[cam].get(f)
            if d is not None:
                s[f] = SEEN_FLAGGED if d.unreliable else SEEN
            elif np.all(np.isfinite(gt[f])):
                camera = make_camera(args.data, cam, f + args.frame_offset, _image_size(args, cam))
                uv = camera.project(gt[f])
                if np.all(np.isfinite(uv)) and 0 <= uv[0] < camera.width and 0 <= uv[1] < camera.height:
                    s[f] = OCCLUDED
        state[cam] = s
    return state


@dataclass
class ObstacleMap:
    origin: np.ndarray          # (2,) м, левый нижний угол сетки
    cell: float
    free: np.ndarray            # (nx, ny) bool — через клетку прошёл хотя бы один «видящий» луч
    blocked: np.ndarray         # (nx, ny) int — сколько «загороженных» лучей прошло через клетку
    blocked_cameras: np.ndarray  # (nx, ny) int — сколько разных камер дали такие лучи

    @property
    def possible(self) -> np.ndarray:
        """Зона, где может быть препятствие (хватает одной камеры)."""
        return (self.blocked > 0) & ~self.free

    @property
    def confirmed(self) -> np.ndarray:
        """Препятствие, в которое упираются лучи хотя бы двух камер."""
        return self.possible & (self.blocked_cameras >= 2)

    def cells_on_segment(self, a: np.ndarray, b: np.ndarray, start: float, stop: float) -> np.ndarray:
        """Клетки (k, 2) на отрезке a -> b между расстояниями start и stop от a (м)."""
        length = np.linalg.norm(b - a)
        if stop <= start:
            return np.zeros((0, 2), int)
        t = np.arange(start, stop, self.cell / 3) / length
        points = a + t[:, None] * (b - a)
        idx = np.floor((points - self.origin) / self.cell).astype(int)
        inside = np.all((idx >= 0) & (idx < self.free.shape), axis=1)
        return np.unique(idx[inside], axis=0)


def build_obstacle_map(cam_xy: dict[int, np.ndarray], truck_xy: np.ndarray, state: dict[int, np.ndarray],
                       lo: np.ndarray, hi: np.ndarray) -> ObstacleMap:
    """Карта по лучам камер (м): сначала все «видящие» лучи вычищают свободное место, потом «загороженные» голосуют."""
    shape = tuple(np.ceil((hi - lo) / CELL_M).astype(int))
    grid = ObstacleMap(lo.copy(), CELL_M, np.zeros(shape, bool), np.zeros(shape, int), np.zeros(shape, int))
    valid = np.all(np.isfinite(truck_xy), axis=1)
    for cam in CAMERAS:
        for f in np.nonzero(valid & (state[cam] <= SEEN_FLAGGED))[0]:
            length = np.linalg.norm(truck_xy[f] - cam_xy[cam][f])
            c = grid.cells_on_segment(cam_xy[cam][f], truck_xy[f], 0.0, length - NEAR_TRUCK_M)
            grid.free[c[:, 0], c[:, 1]] = True
    for cam in CAMERAS:
        touched = np.zeros(shape, bool)
        for f in np.nonzero(valid & (state[cam] == OCCLUDED))[0]:
            length = np.linalg.norm(truck_xy[f] - cam_xy[cam][f])
            c = grid.cells_on_segment(cam_xy[cam][f], truck_xy[f], NEAR_CAMERA_M, length - NEAR_TRUCK_M)
            grid.blocked[c[:, 0], c[:, 1]] += 1
            touched[c[:, 0], c[:, 1]] = True
        grid.blocked_cameras += touched
    return grid


def shadow_start(grid: ObstacleMap, camera_xy: np.ndarray, truck_xy: np.ndarray) -> float:
    """
    Расстояние (м) от камеры, с которого луч к грузовику точно в тени: первое подтверждённое
    препятствие на луче, а если его нет — конец зоны поиска (перед самим грузовиком).
    """
    length = np.linalg.norm(truck_xy - camera_xy)
    for d in np.arange(NEAR_CAMERA_M, length - NEAR_TRUCK_M, grid.cell / 3):
        p = camera_xy + d / length * (truck_xy - camera_xy)
        i, j = np.floor((p - grid.origin) / grid.cell).astype(int)
        if 0 <= i < grid.free.shape[0] and 0 <= j < grid.free.shape[1] and grid.confirmed[i, j]:
            return float(d)
    return max(length - NEAR_TRUCK_M, NEAR_CAMERA_M)


def _annular_sector(center, angle_from, angle_to, r_from, r_to, n=6) -> np.ndarray:
    a = np.linspace(angle_from, angle_to, n)
    outer = center + r_to * np.c_[np.cos(a), np.sin(a)]
    inner = center + r_from * np.c_[np.cos(a[::-1]), np.sin(a[::-1])]
    return np.vstack([outer, inner])


def _fov_angles(camera) -> tuple[float, float]:
    half = np.arctan(camera.c_x / camera.f_x)
    yaw = np.arctan2(camera.forward[1], camera.forward[0])
    return yaw - half, yaw + half


def plot_camera_coverage(out: Path, gt: np.ndarray, args, detections) -> dict[int, np.ndarray]:
    """Вид сверху: секторы камер со светом и тенью, оценённые препятствия, участки траектории по видимости."""
    plt = _style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    upm = args.units_per_meter
    valid = np.all(np.isfinite(gt), axis=1)
    truck = gt[:, :2] / upm
    state = camera_coverage(args, detections, gt)
    frames = np.arange(NUM_VIDEO_FRAMES)
    cameras = {cam: [make_camera(args.data, cam, int(f) + args.frame_offset, _image_size(args, cam)) for f in frames]
               for cam in CAMERAS}
    cam_xy = {cam: np.array([c.position[:2] / upm for c in cameras[cam]]) for cam in CAMERAS}

    every = np.vstack([truck[valid]] + [cam_xy[cam][:1] for cam in CAMERAS])
    lo, hi = every.min(axis=0) - 25, every.max(axis=0) + 25
    radius = max(np.nanmax(np.linalg.norm(truck[valid] - cam_xy[cam][0], axis=1)) for cam in CAMERAS) * 1.05
    grid = build_obstacle_map(cam_xy, truck, state, lo, hi)

    aspect = (hi[1] - lo[1]) / (hi[0] - lo[0])
    cols = 3
    rows = int(np.ceil((len(CAMERAS) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6 * aspect + 2.4), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(CAMERAS) + 1:]:
        ax.set_visible(False)
    extent = (grid.origin[0], grid.origin[0] + grid.free.shape[0] * grid.cell,
              grid.origin[1], grid.origin[1] + grid.free.shape[1] * grid.cell)

    def draw_obstacles(ax, cam=None):
        possible, confirmed = grid.possible, grid.confirmed
        if cam is not None:
            own = _camera_blocked(grid, cam_xy[cam], truck, state[cam])
            possible, confirmed = possible & own, confirmed & own
        rgba = np.zeros(possible.shape + (4,))
        rgba[possible] = [0.55, 0.54, 0.52, 0.28]
        rgba[confirmed] = [0.24, 0.235, 0.224, 0.9]
        ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", extent=extent, zorder=4, interpolation="nearest")

    def base(ax, title):
        ax.plot(truck[valid, 0], truck[valid, 1], color="#d6d5d0", lw=5, solid_capstyle="round", zorder=2)
        for cam in CAMERAS:
            p = cam_xy[cam][0]
            ax.scatter(*p, marker="^", s=70, color=CAMERA_COLORS[cam], edgecolor="white", lw=1.5, zorder=7)
            ax.annotate(f"Cam{cam}", p, xytext=(6, 6), textcoords="offset points", color=INK_2, fontsize=9, zorder=8)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=8)

    def draw_sector(ax, cam, lit_alpha, dim_alpha):
        """Сектор камеры по углам: светлая часть до препятствия, тусклая — за ним (тень)."""
        color = CAMERA_COLORS[cam]
        bins = np.radians(ANGLE_BIN_DEG)
        # углы, покрытые полем зрения хоть на одном кадре (Camera5 поворачивается)
        covered: dict[int, float] = {}            # номер углового бина -> начало тени (м) или inf
        pos = cam_xy[cam][0]
        for f in frames[:: 1 if cam == 5 else NUM_VIDEO_FRAMES]:
            a0, a1 = _fov_angles(cameras[cam][f])
            for k in range(int(np.floor(a0 / bins)), int(np.ceil(a1 / bins))):
                covered.setdefault(k, np.inf)
        for f in np.nonzero(valid & (state[cam] == OCCLUDED))[0]:
            d = shadow_start(grid, cam_xy[cam][f], truck[f])
            ang = np.arctan2(*(truck[f] - cam_xy[cam][f])[::-1])
            k = int(np.floor(ang / bins))
            covered[k] = min(covered.get(k, np.inf), d)
        for k, d in covered.items():
            a0, a1 = k * bins, (k + 1) * bins
            if np.isfinite(d):
                lit = _annular_sector(pos, a0, a1, 0, d)
                dark = _annular_sector(pos, a0, a1, d, radius)
                ax.fill(lit[:, 0], lit[:, 1], color=color, alpha=lit_alpha, lw=0, zorder=0)
                ax.fill(dark[:, 0], dark[:, 1], color="#7a7975", alpha=dim_alpha, lw=0, zorder=0)
            else:
                lit = _annular_sector(pos, a0, a1, 0, radius)
                ax.fill(lit[:, 0], lit[:, 1], color=color, alpha=lit_alpha, lw=0, zorder=0)

    # Обзор: все камеры и все оценённые препятствия
    ax = axes[0]
    seen = sum((state[cam] <= SEEN_FLAGGED).astype(int) for cam in CAMERAS)
    base(ax, f"Обзор: препятствия и сколько камер нашли грузовик\n"
             f"загорожен хотя бы на одной камере: {int((valid & np.any([state[c] == OCCLUDED for c in CAMERAS], 0)).sum())}"
             f" кадров из {int(valid.sum())}")
    for cam in CAMERAS:
        draw_sector(ax, cam, 0.035, 0.0)
    draw_obstacles(ax)
    ramp = {0: ORANGE, 1: "#b7d3f6", 2: "#6da7ec", 3: "#2a78d6", 4: "#184f95", 5: "#0d366b"}
    handles = []
    for k in sorted(ramp):
        m = valid & (seen == k)
        if m.any():
            handles.append(ax.scatter(truck[m, 0], truck[m, 1], s=10, color=ramp[k], lw=0, zorder=5,
                                      label={0: "ни одной камеры", 1: "1 камера"}.get(k, f"{k} камеры")))
    for f in range(0, NUM_VIDEO_FRAMES, 200):
        if valid[f]:
            ax.annotate(f"кадр {f}", truck[f], xytext=(5, -11), textcoords="offset points", fontsize=8,
                        color=INK_2, zorder=8)
    ax.legend(handles=handles, loc="lower left", fontsize=8, markerscale=1.5, title="нашли грузовик",
              title_fontsize=8)

    for ax, cam in zip(axes[1:], CAMERAS):
        s = state[cam]
        in_view = int((s != OUT_OF_VIEW).sum())
        title = f"Cam{cam}: в поле зрения {in_view} кадров, загорожен {int((s == OCCLUDED).sum())}"
        if cam == 5:
            title += "\n(сектор — все направления за время поворота)"
        base(ax, title)
        draw_sector(ax, cam, 0.13, 0.10)
        draw_obstacles(ax, cam)
        for code in (OUT_OF_VIEW, SEEN_FLAGGED, SEEN, OCCLUDED):
            m = valid & (s == code)
            if m.any():
                ax.scatter(truck[m, 0], truck[m, 1], s=12 if code == OCCLUDED else 8, color=COVERAGE_COLORS[code],
                           lw=0, zorder=5 if code != OUT_OF_VIEW else 3)

    handles = [Line2D([], [], ls="", marker="o", ms=7, color=COVERAGE_COLORS[k], label=COVERAGE_LABELS[k])
               for k in (SEEN, SEEN_FLAGGED, OCCLUDED, OUT_OF_VIEW)]
    handles += [Patch(color="#9ec5f4", alpha=0.6, label="сектор камеры: видно"),
                Patch(color="#7a7975", alpha=0.35, label="сектор за препятствием: тень"),
                Patch(color=OBSTACLE_COLOR, label="препятствие (подтверждено 2 камерами)"),
                Patch(color="#8c8a85", alpha=0.35, label="где может быть препятствие (1 камера)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Вид сверху: секторы обзора камер, загораживание и оценка препятствий по лучам "
                 "(положение грузовика — по CSV цели; X вправо, Y вверх, м)",
                 x=0.01, ha="left", fontweight="bold", color=INK_2, fontsize=12)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    fig.savefig(out / "06_camera_coverage_top.png", dpi=120)
    plt.close(fig)
    np.savez(out / "obstacle_map.npz", origin=grid.origin, cell=grid.cell, free=grid.free, blocked=grid.blocked,
             blocked_cameras=grid.blocked_cameras)
    return state


def _camera_blocked(grid: ObstacleMap, cam_xy: np.ndarray, truck_xy: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Клетки, через которые проходили «загороженные» лучи именно этой камеры."""
    mask = np.zeros(grid.free.shape, bool)
    for f in np.nonzero(np.all(np.isfinite(truck_xy), axis=1) & (state == OCCLUDED))[0]:
        length = np.linalg.norm(truck_xy[f] - cam_xy[f])
        c = grid.cells_on_segment(cam_xy[f], truck_xy[f], NEAR_CAMERA_M, length - NEAR_TRUCK_M)
        mask[c[:, 0], c[:, 1]] = True
    return mask
