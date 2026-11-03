import numpy as np

from pipeline.coverage import OCCLUDED, OUT_OF_VIEW, SEEN, build_obstacle_map, shadow_start

FRAMES = 60


def scene():
    """Грузовик едет по y = 0 от x = -60 до 60; стена x ∈ [-10, 10], y ∈ [20, 24] загораживает его от двух камер."""
    truck = np.c_[np.linspace(-60, 60, FRAMES), np.zeros(FRAMES)]
    cams = {1: np.array([-30.0, 80.0]), 2: np.array([30.0, 80.0]), 3: np.array([0.0, -80.0]),
            4: np.array([-200.0, 200.0]), 5: np.array([200.0, 200.0])}

    def blocked(cam, p):
        a, b = cams[cam], p
        t = np.linspace(0, 1, 400)
        q = a + t[:, None] * (b - a)
        return np.any((np.abs(q[:, 0]) <= 10) & (q[:, 1] >= 20) & (q[:, 1] <= 24))

    state = {}
    for cam in cams:
        if cam in (4, 5):
            state[cam] = np.full(FRAMES, OUT_OF_VIEW)
        else:
            state[cam] = np.array([OCCLUDED if blocked(cam, truck[f]) else SEEN for f in range(FRAMES)])
    cam_xy = {cam: np.repeat(cams[cam][None, :], FRAMES, axis=0) for cam in cams}
    return truck, cam_xy, state


def test_obstacle_confirmed_by_two_cameras_is_at_the_wall():
    truck, cam_xy, state = scene()
    assert (state[1] == OCCLUDED).any() and (state[2] == OCCLUDED).any()
    grid = build_obstacle_map(cam_xy, truck, state, np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    idx = np.argwhere(grid.confirmed)
    assert len(idx) > 0
    centers = grid.origin + (idx + 0.5) * grid.cell
    # По x — у стены (с запасом на клетку). По y зона шире стены (y 20–24): полосу между стеной
    # и дорогой не пересекает ни один «видящий» луч, её нельзя вычистить — это предел метода
    assert np.all(np.abs(centers[:, 0]) <= 16)
    assert np.all((centers[:, 1] >= 8) & (centers[:, 1] <= 36))
    assert np.any((centers[:, 1] >= 20) & (centers[:, 1] <= 24))
    # Камера 3 смотрит снизу и видит грузовик всегда: её лучи вычищают пространство под дорогой
    assert not grid.possible[:, : int((0 - grid.origin[1]) / grid.cell)].any()


def test_shadow_starts_at_confirmed_obstacle():
    truck, cam_xy, state = scene()
    grid = build_obstacle_map(cam_xy, truck, state, np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    f = int(np.nonzero(state[1] == OCCLUDED)[0][len(np.nonzero(state[1] == OCCLUDED)[0]) // 2])
    d = shadow_start(grid, cam_xy[1][f], truck[f])
    to_wall = np.linalg.norm(cam_xy[1][f] - truck[f]) * (80 - 24) / 80        # расстояние до верхней грани стены
    assert abs(d - to_wall) < 8
