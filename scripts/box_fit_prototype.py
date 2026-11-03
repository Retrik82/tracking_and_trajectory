"""
ПРОТОТИП: точка кабины без переобучения YOLO.

Грузовик — 3D-коробка L×W×H с опорной точкой в низу передка кабины. Траектория этой точки —
B-сплайн, направление коробки — касательная сплайна. Сплайн и общие габариты подбираются
least_squares (Huber, штраф на рывок) так, чтобы проекция коробки совпала с рамками YOLO
всех камер; рёбра рамки у края кадра не учитываются. CSV цели — только для оценки.

Требует сохранённых детекций и trajectory.npz от scripts/yolo_eval.py (начальное приближение).

Известные ограничения: ширина не определяется (выходит ~6.6 м вместо ~2.5 м), оптимизатор
не сходится за 2000 итераций, жёсткая коробка не описывает складывание фуры в поворотах.

  .venv/bin/python scripts/box_fit_prototype.py --knot-spacing 10 --max-nfev 2000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reconstruction.bspline import _create_open_uniform_knots  # noqa: E402
from reconstruction.data_io import load_target, make_camera  # noqa: E402
from scripts.yolo_eval import load_detections  # noqa: E402

_p = argparse.ArgumentParser(description="Прототип: точка кабины по 3D-коробке")
_p.add_argument("--out", type=Path, default=ROOT / "outputs/yolo_eval", help="папка с детекциями и trajectory.npz")
_p.add_argument("--data", type=Path, default=ROOT / "tmp/project2")
_p.add_argument("--knot-spacing", type=float, default=10.0, help="шаг узлов, кадры")
_p.add_argument("--jerk-weight", type=float, default=1.0)
_p.add_argument("--width-prior", type=float, default=0.0, help="вес априори W = 2.55 м; 0 — без него")
_p.add_argument("--max-nfev", type=int, default=2000)
_args = _p.parse_args()
OUT, DATA = _args.out, _args.data
KNOT_SPACING, JERK_WEIGHT = _args.knot_spacing, _args.jerk_weight
WIDTH_PRIOR, MAX_NFEV = _args.width_prior, _args.max_nfev
MARGIN = 3.0
F_SCALE_PX = 5.0
NF = 1001
UPM = 100.0
FPS = 30.0

dets = load_detections(OUT)
traj = np.load(OUT / "trajectory.npz")
center = traj["spline"]                                  # сплайн по центрам масок (прошлый прогон)

# ---- наблюдения: одна строка на рамку
obs = []
cams = {}
for cam, by_frame in dets.items():
    for f, d in by_frame.items():
        key = (cam, f)
        if cam != 5:
            key = (cam, 0)
        if key not in cams:
            cams[key] = make_camera(DATA, cam, key[1])
        c = cams[key]
        edges = np.array([d.x1, d.x2, d.y1, d.y2])
        valid = np.array([d.x1 > MARGIN, d.x2 < 1920 - MARGIN, d.y1 > MARGIN, d.y2 < 1080 - MARGIN])
        obs.append((f, c, edges, valid, d.confidence))
obs.sort(key=lambda o: o[0])
T = np.array([o[0] for o in obs], float)
Rm = np.stack([o[1].R for o in obs])
Cp = np.stack([o[1].position for o in obs])
FX = np.array([o[1].f_x for o in obs]); FY = np.array([o[1].f_y for o in obs])
EDGES = np.stack([o[2] for o in obs]); VALID = np.stack([o[3] for o in obs]); CONF = np.array([o[4] for o in obs])
print(f"рамок: {len(obs)}, рёбер в работе: {VALID.sum()}")

# ---- сплайн точки кабины
deg = 3
n_cp = int(np.ceil(1000 / KNOT_SPACING)) + deg
knots = _create_open_uniform_knots(n_cp, deg, 0.0, 1000.0)
B = BSpline.design_matrix(T, knots, deg).tocsr()
eye = np.eye(n_cp)
dB = np.stack([BSpline(knots, eye[i], deg).derivative()(T) for i in range(n_cp)], axis=1)

# корни коробки в осях грузовика: x вперёд (0..-L), y вбок (±W/2), z вверх (0..H)
SIGNS = np.array([[x, y, z] for x in (0, 1) for y in (-0.5, 0.5) for z in (0, 1)], float)


def unpack(theta):
    C = theta[:3 * n_cp].reshape(n_cp, 3)
    L, W, H = theta[3 * n_cp:]
    return C, L, W, H


def axes(vel):
    h = vel[:, :2]
    hn = np.linalg.norm(h, axis=1, keepdims=True)
    fwd = vel / np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9)
    side = np.c_[-h[:, 1:2] / np.maximum(hn, 1e-9), h[:, 0:1] / np.maximum(hn, 1e-9), np.zeros((len(h), 1))]
    up = np.cross(fwd, side)
    up *= np.sign(up[:, 2:3] + 1e-12)
    return fwd, side, up


def project_box(theta, t_B, t_dB, Rm_, Cp_, FX_, FY_):
    C, L, W, H = unpack(theta)
    p = t_B @ C
    v = t_dB @ C
    fwd, side, up = axes(v)
    corners = (p[:, None, :] - SIGNS[None, :, 0:1] * L * fwd[:, None, :]
               + SIGNS[None, :, 1:2] * W * side[:, None, :] + SIGNS[None, :, 2:3] * H * up[:, None, :])
    cam = np.einsum("nij,nkj->nki", Rm_, corners - Cp_[:, None, :])
    z = np.maximum(cam[..., 2], 1.0)
    u = 960 + FX_[:, None] * cam[..., 0] / z
    vv = 540 + FY_[:, None] * cam[..., 1] / z
    return np.c_[u.min(1), u.max(1), vv.min(1), vv.max(1)], p


D3 = np.diff(np.eye(n_cp), n=3, axis=0)


def residuals(theta):
    box, _ = project_box(theta, B, dB, Rm, Cp, FX, FY)
    r = (box - EDGES) * np.sqrt(CONF)[:, None]
    r = np.where(VALID, r, 0.0).ravel()
    C, L, W, H = unpack(theta)
    jerk = (D3 @ C).ravel() * JERK_WEIGHT / UPM * 10
    prior = [(W - 255.0) / 30.0 * WIDTH_PRIOR] if WIDTH_PRIOR else []
    return np.r_[r, jerk, prior]


# ---- начальное приближение: центр + половина длины вперёд, вниз на половину высоты
L0, W0, H0 = 1600.0, 250.0, 400.0
tt = np.arange(NF, dtype=float)
ok = np.all(np.isfinite(center), axis=1)
cen = center.copy()
for k in range(3):
    cen[:, k] = np.interp(tt, tt[ok], center[ok, k])
vel0 = np.gradient(cen, axis=0)
fwd0, _, up0 = axes(vel0)
p0 = cen + L0 / 2 * fwd0 - H0 / 2 * up0
Bfull = BSpline.design_matrix(tt, knots, deg).toarray()
C0 = np.linalg.lstsq(Bfull, p0, rcond=None)[0]
theta0 = np.r_[C0.ravel(), L0, W0, H0]

# ---- разреженность якобиана
n_obs = len(obs)
S = lil_matrix((4 * n_obs + D3.shape[0] * 3 + (1 if WIDTH_PRIOR else 0), len(theta0)), dtype=np.int8)
if WIDTH_PRIOR:
    S[-1, 3 * n_cp + 1] = 1
Bd = B.toarray()
for i in range(n_obs):
    cps = np.nonzero((Bd[i] != 0) | (dB[i] != 0))[0]
    for cp in cps:
        S[4 * i:4 * i + 4, 3 * cp:3 * cp + 3] = 1
    S[4 * i:4 * i + 4, 3 * n_cp:] = 1
for j in range(D3.shape[0]):
    for cp in np.nonzero(D3[j])[0]:
        S[4 * n_obs + 3 * j:4 * n_obs + 3 * j + 3, 3 * cp:3 * cp + 3] = 1

res = least_squares(residuals, theta0, jac_sparsity=S, loss="huber", f_scale=F_SCALE_PX, x_scale="jac",
                    max_nfev=MAX_NFEV, verbose=0)
C, L, W, H = unpack(res.x)
print(f"оптимизация: {res.message} nfev={res.nfev}")
print(f"габариты из данных: L={L / UPM:.2f} м, W={W / UPM:.2f} м, H={H / UPM:.2f} м")
box, _ = project_box(res.x, B, dB, Rm, Cp, FX, FY)
edge_err = np.abs(box - EDGES)[VALID]
print(f"ошибка рёбер рамки, px: медиана {np.median(edge_err):.1f}, p95 {np.percentile(edge_err, 95):.1f}")

# ---- сравнение с CSV цели
spl = BSpline(knots, C, deg)
P = spl(tt)
target = load_target(DATA)
gt = np.array([target.position(f) for f in range(NF)])
valid = np.all(np.isfinite(gt), axis=1)
err = np.linalg.norm(P - gt, axis=1)[valid] / UPM


def st(e):
    return f"медиана {np.median(e):.2f} / p95 {np.percentile(e, 95):.2f} / RMSE {np.sqrt(np.mean(e ** 2)):.2f} / макс {e.max():.2f}"


print(f"точка кабины vs CSV цели, м (все {valid.sum()} кадров): {st(err)}")
gt_vel = np.gradient(gt, axis=0)
fwd, side, up = axes(np.where(np.isfinite(gt_vel), gt_vel, 1.0))
diff = (P - gt) / UPM
for name, ax in (("вдоль", fwd), ("поперёк", side), ("вертикаль", up)):
    comp = np.sum(diff * ax, axis=1)
    m = valid & np.all(np.isfinite(gt_vel), axis=1) & (np.linalg.norm(gt_vel, axis=1) > 3.3)
    print(f"  {name}: среднее {np.mean(comp[m]):+.2f} м, |медиана| {np.median(np.abs(comp[m])):.2f} м")
speed = np.linalg.norm(spl.derivative()(tt), axis=1) * FPS / UPM
acc = np.linalg.norm(spl.derivative(2)(tt), axis=1) * FPS ** 2 / UPM
gspeed = np.linalg.norm(np.gradient(np.where(valid[:, None], gt, np.nan), axis=0), axis=1) * FPS / UPM
se = np.abs(speed - gspeed)[np.isfinite(gspeed)]
print(f"скорость: макс {speed.max():.1f} м/с (цель ~32), ошибка: медиана {np.median(se):.2f}, p95 {np.percentile(se, 95):.2f} м/с; макс ускорения {acc.max():.1f} м/с²")
np.savez(OUT / "box_fit_result.npz", cab=P, L=L, W=W, H=H, err=np.linalg.norm(P - gt, axis=1) / UPM)
