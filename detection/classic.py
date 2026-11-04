"""
Детектор без нейросети: вычитание фона по известным позам камер.

Почему это работает в этой сцене. Камеры 1–4 неподвижны, Camera5 только вращается вокруг
своего центра (кадры 565–959), и её поворот известен из CSV. Деревья и рельеф не двигаются,
поэтому всё, что отличается от фона, — грузовик (и его тень).

Строго причинно: фон на кадре t построен только по кадрам ≤ t.

1. Фон — бегущее среднее кадров в половинном разрешении. У вращающейся камеры фон хранится
   панорамой по направлениям в мире (азимут × угол места): центр камеры не двигается, поэтому
   каждый пиксель кадра — это направление, и фон для любого поворота берётся из панорамы одной
   интерполяцией. Перенос фона от кадра к кадру гомографией размывал бы его с каждым кадром,
   и края деревьев начинали бы «отличаться от фона». Неувиденные места панорамы заполняются
   текущим кадром.
2. Передний план — пиксели, где цвет отличается от фона больше порога; морфология склеивает
   куски кузова, разрезанные деревом. Тень остаётся в маске: проверенные признаки тени
   («темнее при том же оттенке», «темнее и синее») вырезали и серо-голубой кузов на снегу.
3. Фон обновляется выборочно: под передним планом не обновляется (грузовик не «впитывается»
   в фон), а пятно, неподвижное целиком, быстро впитывается. Так убирается «призрак» на месте,
   где грузовик стоял в первом кадре. Попиксельно так делать нельзя: середина однотонного кузова
   от кадра к кадру не меняется, впитывалась бы в фон и оставляла след за грузовиком.
4. Рамка пятна строится по пикселям, менявшимся за последние кадры (края едущего грузовика),
   поэтому неподвижный «призрак», слившийся с грузовиком, её не растягивает. Из пятен берётся
   то, что ближе к прошлой рамке (а без неё — самое большое).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detection.schema import Detection
from detection.yolo import FrameResult


@dataclass
class ClassicSettings:
    scale: float = 0.5                  # обработка в уменьшенном кадре
    diff_threshold: float = 28.0        # порог отличия от фона, уровни 0–255 (максимум по каналам)
    motion_threshold: float = 8.0       # порог «пиксель изменился с прошлого кадра»
    alpha_background: float = 0.03      # скорость обновления фона там, где переднего плана нет
    alpha_still: float = 0.15           # скорость впитывания «призрака» (неподвижного пятна)
    still_frames: int = 12              # столько кадров без изменений — передний план неподвижен
    ghost_still_fraction: float = 0.97  # пятно, неподвижное на такой доле пикселей, — «призрак»
    active_frames: int = 6              # «менялся недавно»: не позже стольких кадров назад
    min_active_fraction: float = 0.05   # меньше — объект стоит, рамка по всему пятну
    provisional_frames: int = 150       # панорама: участок, увиденный впервые недавно, — фон под вопросом
    close_px: int = 15                  # закрытие маски, пиксели уменьшенного кадра
    group_px: int = 25                  # пятна ближе этого объединяются в один объект
    min_mask_area_px: float = 400.0     # минимальная площадь, пиксели полного кадра
    edge_margin_px: float = 2.0
    warmup_frames: int = 3              # первые кадры только набирают фон


class ClassicDetector:
    def __init__(self, settings: ClassicSettings | None = None):
        self.settings = settings or ClassicSettings()

    def detect_video(self, video_path: Path, camera: int, rotations=None, K: np.ndarray | None = None
                     ) -> Iterator[FrameResult]:
        """
        rotations — последовательность матриц поворота мир→камера по кадрам (None — камера
        неподвижна); K — внутренние параметры полного кадра.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        state = _State(self.settings, K)
        frame = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                R = rotations[frame] if rotations is not None and frame < len(rotations) else None
                yield state.step(image, camera, frame, R)
                frame += 1
        finally:
            cap.release()


class _State:
    def __init__(self, s: ClassicSettings, K: np.ndarray | None):
        self.s = s
        self.K_small = None if K is None else np.diag([s.scale, s.scale, 1.0]) @ K
        self.background = None          # float32 (h, w, 3)
        self.prev = None                # прошлый уменьшенный кадр, float32
        self.still = None               # сколько кадров подряд пиксель не менялся
        self.prev_R = None
        self.prev_box = None            # прошлая рамка в уменьшенном кадре
        self.pano = None                # фон вращающейся камеры
        self.provisional = None         # пиксели кадра на недавно увиденных участках панорамы
        self.frames_seen = 0

    def _warp(self, H: np.ndarray, shape) -> np.ndarray:
        """Перенос прошлого кадра и счётчиков в текущий вид (одна интерполяция, без накопления)."""
        h, w = shape[:2]
        self.prev = cv2.warpPerspective(self.prev, H, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        self.still = cv2.warpPerspective(self.still, H, (w, h), flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if self.prev_box is not None:
            x1, y1, x2, y2 = self.prev_box
            corners = cv2.perspectiveTransform(np.array([[[x1, y1], [x2, y1], [x1, y2], [x2, y2]]],
                                                        dtype=np.float32), H)[0]
            self.prev_box = (*corners.min(axis=0), *corners.max(axis=0))

    def step(self, image: np.ndarray, camera: int, frame: int, R: np.ndarray | None) -> FrameResult:
        s = self.s
        h_full, w_full = image.shape[:2]
        small = cv2.resize(image, None, fx=s.scale, fy=s.scale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)

        use_pano = R is not None and self.K_small is not None
        if self.prev is None:
            if use_pano:
                self.pano = Panorama(self.K_small, small.shape, R)
                self.pano.update(small, R, np.ones(small.shape[:2], np.float32))
            self.background = small.copy()
            self.prev = small.copy()
            self.still = np.zeros(small.shape[:2], np.uint16)
            self.prev_R = R
            self.frames_seen = 1
            return FrameResult(None, None)

        if use_pano:
            # Прошлый кадр — в текущий вид гомографией чистого поворота H = K·R_t·R_{t-1}ᵀ·K⁻¹
            if not np.allclose(R, self.prev_R, atol=1e-9):
                H = self.K_small @ R @ self.prev_R.T @ np.linalg.inv(self.K_small)
                self._warp(H, small.shape)
                prev_nan = np.isnan(self.prev[..., 0])
                self.prev[prev_nan] = small[prev_nan]
            self.background, age = self.pano.view(R, fallback=small)
            self.provisional = age < s.provisional_frames
        self.prev_R = R

        diff = np.abs(small - self.background).max(axis=2)
        changed = diff > s.diff_threshold

        mask = changed.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.close_px, s.close_px)))

        moving = np.abs(small - self.prev).max(axis=2) > s.motion_threshold
        self.still = np.where(moving, 0, np.minimum(self.still.astype(np.int32) + 1, 60000)).astype(np.uint16)

        # Пятна переднего плана. «Призрак» (место, где объект был, а фон ещё старый) не шевелится
        # нигде, а у едущего грузовика всегда меняются края — даже если середина однотонная
        # и от кадра к кадру одинакова. Поэтому впитывается только пятно целиком неподвижное.
        candidates, ghosts = self._groups(mask)
        # Фон не обновляется ни под объектом, ни рядом с ним: иначе он впитывает объект
        frozen = (mask > 0) | cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        alpha = np.where(frozen, 0.0, s.alpha_background).astype(np.float32)
        for member in ghosts:
            alpha[member] = s.alpha_still
        # Вращающаяся камера впервые видит участок фона, а там как раз грузовик — в панораму
        # попадает «призрак», примыкающий к грузовику (в одно пятно с ним, как пятно не
        # впитывается). На недавно увиденных участках неподвижное впитывается попиксельно.
        if use_pano:
            alpha[self.provisional & (self.still >= s.still_frames)] = s.alpha_still
        if use_pano:
            self.pano.update(small, R, alpha)
        else:
            self.background += alpha[..., None] * (small - self.background)
        self.prev = small
        self.frames_seen += 1
        if self.frames_seen <= s.warmup_frames:
            return FrameResult(None, None)
        return self._pick(candidates, camera, frame, w_full, h_full)

    def _groups(self, mask: np.ndarray):
        s = self.s
        min_area_small = s.min_mask_area_px * s.scale ** 2
        grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.group_px, s.group_px)))
        n, labels = cv2.connectedComponents(grouped)
        candidates, ghosts = [], []
        for k in range(1, n):
            member = (labels == k) & (mask > 0)
            area = int(member.sum())
            if area == 0:
                continue
            if np.mean(self.still[member] >= s.still_frames) >= s.ghost_still_fraction:
                ghosts.append(member)
                continue
            if area < min_area_small:
                continue
            # Рамка — по пикселям, менявшимся за последние кадры. У едущего грузовика меняются оба
            # края, передний и задний, а однотонная середина кузова лежит между ними и попадает в
            # рамку сама. «Призрак» позади (место, где грузовик был, когда фон запоминался) давно
            # не меняется — в рамку не попадает, даже если слился с грузовиком в одно пятно.
            active = member & (self.still < s.active_frames)
            if active.sum() >= s.min_active_fraction * area:
                ys, xs = np.nonzero(active)
                box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                inside = np.zeros_like(member)
                inside[box[1]:box[3], box[0]:box[2]] = True
                member = member & inside
                area = int(member.sum())
            else:
                ys, xs = np.nonzero(member)
                box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            candidates.append((member, area, box))
        return candidates, ghosts

    def _pick(self, candidates, camera: int, frame: int, w_full: int, h_full: int) -> FrameResult:
        s = self.s
        if not candidates:
            self.prev_box = None
            return FrameResult(None, None)

        if self.prev_box is not None:
            px = (self.prev_box[0] + self.prev_box[2]) / 2
            py = (self.prev_box[1] + self.prev_box[3]) / 2
            size = max(self.prev_box[2] - self.prev_box[0], self.prev_box[3] - self.prev_box[1], 1.0)

            def score(c):
                b = c[2]
                dist = np.hypot((b[0] + b[2]) / 2 - px, (b[1] + b[3]) / 2 - py) / size
                return iou(b, self.prev_box) - 0.5 * dist + 0.1 * np.log(c[1])
            member, area, box = max(candidates, key=score)
        else:
            member, area, box = max(candidates, key=lambda c: c[1])
        self.prev_box = tuple(float(v) for v in box)

        inv = 1.0 / s.scale
        x1, y1, x2, y2 = (float(v) * inv for v in box)
        ys, xs = np.nonzero(member)
        cu, cv_ = (float(xs.mean()) + 0.5) * inv, (float(ys.mean()) + 0.5) * inv
        area_full = float(area) * inv ** 2
        contours, _ = cv2.findContours(member.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) * inv if contours else None

        # «Уверенность» — заполненность рамки маской: у грузовика 0.6–0.9, у обрывков меньше
        fill = area / max((box[2] - box[0]) * (box[3] - box[1]), 1)
        m = s.edge_margin_px
        edge_clipped = bool(x1 <= m or y1 <= m or x2 >= w_full - m or y2 >= h_full - m)
        too_small = bool(area_full < s.min_mask_area_px)
        unreliable = edge_clipped or too_small or fill < 0.3
        det = Detection(
            camera=camera, frame=frame, confidence=float(min(fill, 1.0)),
            x1=x1, y1=y1, x2=x2, y2=y2,
            point_u=cu, point_v=cv_, point_source="mask_centroid",
            mask_area_px=area_full, mask_centroid_u=cu, mask_centroid_v=cv_,
            num_candidates=len(candidates),
            edge_clipped=edge_clipped, too_small=too_small, unreliable=unreliable,
        )
        return FrameResult(det, polygon)


class Panorama:
    """
    Фон вращающейся камеры по направлениям в мире: равнопромежуточная развёртка
    (азимут 0–360°, угол места в пределах, которые камера видит при данном наклоне).
    Разрешение — как у кадра: f пикселей на радиан.
    """

    def __init__(self, K: np.ndarray, shape, R0: np.ndarray, margin_deg: float = 3.0):
        self.K, self.K_inv = K, np.linalg.inv(K)
        h, w = shape[:2]
        self.h, self.w = h, w
        self.ppr = float(K[0, 0])                       # пикселей на радиан
        us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        self.rays_cam = np.stack([us, vs, np.ones_like(us)], axis=-1) @ self.K_inv.T   # (h, w, 3)
        _, el = self._angles(self.rays_cam @ R0)
        margin = np.radians(margin_deg)
        self.el_max = float(el.max()) + margin
        self.el_min = float(el.min()) - margin
        self.W = int(np.ceil(2 * np.pi * self.ppr))
        self.H = int(np.ceil((self.el_max - self.el_min) * self.ppr))
        self.image = np.zeros((self.H, self.W, 3), np.float32)
        self.seen = np.zeros((self.H, self.W), np.float32)
        self.age = np.zeros((self.H, self.W), np.float32)     # сколько кадров участок виден

    @staticmethod
    def _angles(d: np.ndarray):
        az = np.arctan2(d[..., 1], d[..., 0])
        el = np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1]))
        return az, el

    def _pano_xy(self, R: np.ndarray):
        """Координаты в панораме для каждого пикселя текущего кадра (лучи в мире = Rᵀ·K⁻¹·[u v 1])."""
        az, el = self._angles(self.rays_cam @ R)
        x = ((az + np.pi) * self.ppr) % self.W
        y = (self.el_max - el) * self.ppr
        return x.astype(np.float32), y.astype(np.float32)

    def view(self, R: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        x, y = self._pano_xy(R)
        # Шов на 360°: дублируем первый столбец справа, чтобы интерполяция не рвалась
        img = np.concatenate([self.image, self.image[:, :1]], axis=1)
        seen = np.concatenate([self.seen, self.seen[:, :1]], axis=1)
        bg = cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ok = cv2.remap(seen, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.99
        bg[~ok] = fallback[~ok]
        age = cv2.remap(np.concatenate([self.age, self.age[:, :1]], axis=1), x, y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        age[~ok] = 0
        return bg, age

    def update(self, small: np.ndarray, R: np.ndarray, alpha: np.ndarray) -> None:
        """Подмешать кадр в панораму с весами alpha (в координатах кадра); новое — целиком."""
        x, y = self._pano_xy(R)
        # Полоса панорамы, которую видит кадр (по азимуту — с учётом шва)
        cx = float(x[self.h // 2, self.w // 2])
        half = int(np.ceil(np.abs(((x - cx + self.W / 2) % self.W) - self.W / 2).max())) + 2
        cols = (int(cx) + np.arange(-half, half + 1)) % self.W
        y0, y1 = max(int(np.floor(y.min())) - 1, 0), min(int(np.ceil(y.max())) + 2, self.H)
        # Для пикселей полосы — где они в кадре: направление → K·R·d
        az = cols.astype(np.float64) / self.ppr - np.pi
        el = self.el_max - np.arange(y0, y1, dtype=np.float64) / self.ppr
        az_g, el_g = np.meshgrid(az, el)
        d = np.stack([np.cos(el_g) * np.cos(az_g), np.cos(el_g) * np.sin(az_g), np.sin(el_g)], axis=-1)
        p = d @ (self.K @ R).T
        z = p[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(z > 0, p[..., 0] / z, -1).astype(np.float32)
            v = np.where(z > 0, p[..., 1] / z, -1).astype(np.float32)
        frame_px = cv2.remap(small, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = cv2.remap(alpha.astype(np.float32), u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=-1)
        inside = (u >= 0) & (v >= 0) & (u <= self.w - 1) & (v <= self.h - 1)
        tile = self.image[y0:y1][:, cols]
        seen = self.seen[y0:y1][:, cols]
        new = inside & (seen < 0.5)
        a = np.where(new, 1.0, np.where(inside, np.maximum(a, 0), 0)).astype(np.float32)
        tile += a[..., None] * (frame_px - tile)
        seen = np.where(inside, 1.0, seen)
        self.image[y0:y1, cols] = tile
        self.seen[y0:y1, cols] = seen
        self.age[y0:y1, cols] = self.age[y0:y1][:, cols] + inside


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
