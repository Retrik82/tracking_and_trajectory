"""Детектор без нейросети: вычитание фона на синтетических кадрах."""

import numpy as np

from detection.classic import ClassicSettings, Panorama, _State


def _scene(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, size=(h, w, 3)).astype(np.uint8)


def test_moving_box_is_found_and_not_absorbed():
    background = _scene()
    state = _State(ClassicSettings(scale=1.0, min_mask_area_px=50, warmup_frames=2), K=None)
    det = None
    for frame in range(60):
        img = background.copy()
        x = 20 + 3 * frame
        img[100:140, x:x + 60] = 250          # однотонный «кузов»: середина от кадра к кадру одинакова
        det = state.step(img, camera=1, frame=frame, R=None).detection
    assert det is not None
    assert abs(det.x1 - x) <= 3 and abs(det.x2 - (x + 60)) <= 3
    assert abs(det.y1 - 100) <= 3 and abs(det.y2 - 140) <= 3


def test_ghost_from_first_frame_disappears():
    """Объект стоит в первом кадре, потом уезжает: место, где он стоял, не должно остаться объектом."""
    background = _scene(seed=1)
    state = _State(ClassicSettings(scale=1.0, min_mask_area_px=50, warmup_frames=2), K=None)
    det = None
    for frame in range(80):
        img = background.copy()
        x = 20 + 4 * frame if frame >= 5 else 20
        if x + 40 < img.shape[1]:
            img[100:140, x:x + 40] = 250
        det = state.step(img, camera=1, frame=frame, R=None).detection
    assert det is None or det.x1 > 200


def test_panorama_view_matches_frame_after_rotation():
    """Фон, сохранённый в панораме, для повёрнутой камеры совпадает с тем, что она видит."""
    h, w = 120, 160
    K = np.array([[150.0, 0, w / 2], [0, 150.0, h / 2], [0, 0, 1]])

    def R_yaw(deg):
        a = np.radians(deg)
        forward = np.array([np.cos(a), np.sin(a), 0.0])
        right = np.array([np.sin(a), -np.cos(a), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        return np.vstack([right, down, forward])

    # Мир: яркость зависит только от азимута — видим, правильно ли панорама кладёт направления
    def render(R):
        us, vs = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        rays = np.stack([us, vs, np.ones_like(us)], axis=-1) @ np.linalg.inv(K).T @ R
        az = np.degrees(np.arctan2(rays[..., 1], rays[..., 0]))
        img = (128 + 100 * np.sin(np.radians(az) * 8)).astype(np.float32)
        return np.repeat(img[..., None], 3, axis=2)

    pano = Panorama(K, (h, w), R_yaw(0))
    for deg in range(0, 31, 2):
        pano.update(render(R_yaw(deg)), R_yaw(deg), np.full((h, w), 0.5, np.float32))
    view, age = pano.view(R_yaw(15), fallback=np.zeros((h, w, 3), np.float32))
    truth = render(R_yaw(15))
    inner = (slice(5, -5), slice(5, -5))
    assert np.median(np.abs(view[inner] - truth[inner])) < 3.0
    assert age[inner].min() > 0
