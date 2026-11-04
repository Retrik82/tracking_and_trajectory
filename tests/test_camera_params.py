"""
Параметры камер: чтение CSV (позиция, углы, фокус в мм), разрешение каждого видео, пересчёт фокуса
в пиксели через постоянную матрицу 23.76 × 13.365 мм и передача параметров по конвейеру.
"""

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pytest

from detection.schema import Detection
from pipeline.config import StreamConfig
from reconstruction.cab_model import DEFAULT_DIMENSIONS, box_corners
from reconstruction.camera import (
    SENSOR_HEIGHT_MM,
    SENSOR_WIDTH_MM,
    Camera,
    CameraCsvError,
    FrameNotFoundError,
    load_camera_poses,
    pixel_focal_lengths,
    read_camera_csv,
)
from reconstruction.data_io import (
    VideoResolutionError,
    camera_csv_path,
    detections_image_sizes,
    find_video,
    frame_resolution,
    make_camera,
    probe_video_resolution,
    read_focal_length_mm,
    resolve_image_sizes,
    write_detections_image_sizes,
)
from reconstruction.triangulation import triangulate

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE2_DIR = Path(os.environ.get("PELENG_SCENE2_DIR", REPO_ROOT / "data"))
requires_scene2 = pytest.mark.skipif(
    not (SCENE2_DIR / "object_transform_CineCameraActor9.csv").exists(),
    reason=f"нет данных сцены 2 в {SCENE2_DIR} (задайте PELENG_SCENE2_DIR)",
)

HEADER2 = "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal_Length_mm"


def write_csv(path: Path, text: str, encoding: str = "cp1251") -> Path:
    path.write_bytes(text.replace("\n", "\r\n").encode(encoding))
    read_camera_csv.cache_clear()          # тот же путь в разных тестах — свежее чтение
    return path


def write_video(path: Path, width: int, height: int, frames: int = 3, fps: float = 30.0) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        img = np.zeros((height, width, 3), np.uint8)
        img[:, : (i * 7) % width] = (0, 120, 255)
        writer.write(img)
    writer.release()
    return path


def scene_csv(data: Path, cam: int, rows: list[tuple], focal_mm=None) -> Path:
    """CSV камеры формата сцены 2 (фокус колонкой); rows — (кадр, x, y, z, roll, pitch, yaw[, фокус])."""
    lines = [HEADER2]
    for r in rows:
        f = r[7] if len(r) > 7 else focal_mm
        lines.append(";".join(str(v) for v in (*r[:7], f)))
    return write_csv(camera_csv_path(data, cam), "\n".join(lines) + "\n")


# ---------------------------------------------------------------- CSV: формат и значения

def test_reads_position_angles_and_focal_from_column_format(tmp_path):
    """Сцена 2: Frame;Loc_X;…;Rot_Yaw;Focal_Length_mm, фокус в каждой строке, углы в градусах."""
    p = write_csv(tmp_path / "cam.csv", HEADER2 + "\n"
                  "0;-28487.3434;5124.9391;1200.0;0.0;1.6;-37.6;35.0\n"
                  "1;-28487.3434;5124.9391;1200.0;-0.6776;-17.514;139.3096;35.0\n")
    data = read_camera_csv(str(p))
    assert data.focal_source == "column" and data.issues == ()
    (x, y, z), roll, pitch, yaw = data.poses[0]
    assert (x, y, z) == (-28487.3434, 5124.9391, 1200.0)
    assert (roll, pitch, yaw) == pytest.approx((0.0, math.radians(1.6), math.radians(-37.6)))
    assert data.poses[1][1:] == pytest.approx(tuple(math.radians(v) for v in (-0.6776, -17.514, 139.3096)))
    assert data.focal_length_mm(0) == 35.0 and read_focal_length_mm(p) == 35.0
    assert load_camera_poses(str(p)) == data.poses           # прежний API — те же позы


def test_reads_focal_from_header_pair_format(tmp_path):
    """Сцена 1: «…;Rot_Yaw;;;Focal length;20» в заголовке, строки — без фокуса."""
    p = write_csv(tmp_path / "cam.csv", "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;20\n"
                  "0;1;2;3;0;10;20\n1;1;2;3;0;10;21\n")
    data = read_camera_csv(str(p))
    assert data.focal_source == "header"
    assert data.focal_mm == {0: 20.0, 1: 20.0}
    assert data.poses[1][3] == pytest.approx(math.radians(21))


def test_columns_found_by_name_not_position_and_comma_decimals(tmp_path):
    p = write_csv(tmp_path / "cam.csv", "Focal_Length_mm;Rot_Yaw;Rot_Pitch;Rot_Roll;Loc_Z;Loc_Y;Loc_X;Frame\n"
                  "23,5;30;-10;1,5;300;200;100;7\n")
    data = read_camera_csv(str(p))
    (x, y, z), roll, pitch, yaw = data.poses[7]
    assert (x, y, z) == (100.0, 200.0, 300.0)
    assert (math.degrees(roll), math.degrees(pitch), math.degrees(yaw)) == pytest.approx((1.5, -10.0, 30.0))
    assert data.focal_mm[7] == 23.5


def test_utf8_bom_file_is_read(tmp_path):
    p = write_csv(tmp_path / "cam.csv", HEADER2 + "\n0;1;2;3;0;0;0;35\n", encoding="utf-8-sig")
    assert read_camera_csv(str(p)).focal_mm == {0: 35.0}


def test_missing_pose_column_is_clear_error(tmp_path):
    p = write_csv(tmp_path / "cam.csv", "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Focal_Length_mm\n0;1;2;3;0;0;35\n")
    with pytest.raises(CameraCsvError, match="нет колонок rot_yaw"):
        read_camera_csv(str(p))


def test_missing_focal_is_clear_error(tmp_path):
    p = write_csv(tmp_path / "cam.csv", "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw\n0;1;2;3;0;0;0\n")
    with pytest.raises(CameraCsvError, match="нет фокусного расстояния"):
        read_camera_csv(str(p))


def test_focal_given_twice_is_ambiguous(tmp_path):
    p = write_csv(tmp_path / "cam.csv", HEADER2 + ";;Focal length;35\n0;1;2;3;0;0;0;35\n")
    with pytest.raises(CameraCsvError, match="несколько раз"):
        read_camera_csv(str(p))


def test_missing_file_and_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="нет CSV камеры"):
        read_camera_csv(str(tmp_path / "nope.csv"))
    with pytest.raises(CameraCsvError, match="пустой"):
        read_camera_csv(str(write_csv(tmp_path / "empty.csv", "")))


def test_invalid_rows_are_skipped_and_reported_not_guessed(tmp_path, caplog):
    p = write_csv(tmp_path / "cam.csv", HEADER2 + "\n"
                  "0;1;2;3;0;0;0;35\n"
                  "1;авг.13;2;3;0;0;0;35\n"          # Excel превратил число в дату
                  "2;1;2;3;0;0;0;\n"                 # нет фокуса
                  "3;1;2;3;0;0\n"                    # неполная строка
                  "4;1;2;3;0;0;0;0\n"                # фокус 0 мм
                  "5;1;2;3;0;0;0;-35\n"              # отрицательный фокус
                  "6;1;2;nan;0;0;0;35\n"             # не число
                  "x;1;2;3;0;0;0;35\n"               # номер кадра
                  "7;1;2;3;0;0;0;35\n"
                  "7;9;9;9;0;0;0;35\n"               # повтор кадра: какая поза верна — неизвестно
                  "8;1;2;3;0;0;0;35\n")
    with caplog.at_level(logging.WARNING, logger="peleng"):
        data = read_camera_csv(str(p))
    assert sorted(data.poses) == [0, 8]
    assert len(data.issues) == 8                 # кадры 1–6, строка «x», повтор кадра 7
    text = " | ".join(data.issues)
    assert "кадр 1: нечисловые или пустые значения в Loc_X" in text
    assert "кадр 2: нечисловые или пустые значения в Focal_Length_mm" in text
    assert "кадр 3: неполная строка" in text and "кадр 4: фокусное расстояние 0" in text
    assert "кадр 7: повторяется" in text and "номер кадра 'x'" in text
    assert "пропущено строк" in caplog.text
    with pytest.raises(FrameNotFoundError):
        Camera.from_csv(str(p), 1, 1920, 1080)


def test_file_without_valid_rows_is_error(tmp_path):
    p = write_csv(tmp_path / "cam.csv", HEADER2 + "\n0;a;b;c;0;0;0;35\n")
    with pytest.raises(CameraCsvError, match="нет ни одной корректной строки"):
        read_camera_csv(str(p))


def test_parameters_are_bound_to_the_right_camera(tmp_path):
    """Файл камеры N — object_transform_CineCameraActor{N}.csv; Actor1 и Actor10 не путаются."""
    scene_csv(tmp_path, 1, [(0, 100, 0, 0, 0, 0, 10)], focal_mm=25)
    scene_csv(tmp_path, 10, [(0, 1000, 0, 0, 0, 0, 20)], focal_mm=60)
    a = make_camera(tmp_path, 1, 0, (1920, 1080))
    b = make_camera(tmp_path, 10, 0, (1920, 1080))
    assert a.position[0] == 100 and a.focal_length_mm == 25 and a.yaw == pytest.approx(math.radians(10))
    assert b.position[0] == 1000 and b.focal_length_mm == 60 and b.yaw == pytest.approx(math.radians(20))


def test_focal_changing_per_frame_zoom(tmp_path):
    p = scene_csv(tmp_path, 1, [(0, 0, 0, 0, 0, 0, 0, 25), (1, 0, 0, 0, 0, 0, 0, 50)])
    with pytest.raises(CameraCsvError, match="меняется по кадрам"):
        read_focal_length_mm(p)
    assert read_focal_length_mm(p, frame=1) == 50
    c0, c1 = make_camera(tmp_path, 1, 0, (1920, 1080)), make_camera(tmp_path, 1, 1, (1920, 1080))
    assert c1.f_x == pytest.approx(2 * c0.f_x)
    assert c0.update_from_csv_frame(p, 1) and c0.focal_length_mm == 50 and c0.f_x == pytest.approx(c1.f_x)
    np.testing.assert_allclose(c0.K, c1.K)


@requires_scene2
def test_real_scene2_csv_files():
    """Реальные CSV сцены 2: 1933 строки (кадры 0–1932), фокус колонкой, без испорченных строк."""
    focal = {1: 35.0, 2: 60.0, 3: 35.0, 4: 50.0, 5: 25.0, 6: 35.0, 7: 35.0, 8: 35.0, 9: 35.0}
    for cam, f_mm in focal.items():
        data = read_camera_csv(str(camera_csv_path(SCENE2_DIR, cam)))
        assert data.focal_source == "column" and not data.issues
        assert sorted(data.poses) == list(range(1933))
        assert set(data.focal_mm.values()) == {f_mm}
    first = read_camera_csv(str(camera_csv_path(SCENE2_DIR, 1))).poses[0]
    assert first[0] == (-28487.3434, 5124.9391, 1200.0)
    assert np.degrees(first[1:]) == pytest.approx((0.0, 1.6, -37.6))
    assert (SCENE2_DIR / "matrix_size.txt").read_text().strip() == "23,76 mm * 13,365 mm"


# ---------------------------------------------------------------- матрица, фокус, K

@pytest.mark.parametrize("width,height", [(1920, 1080), (1280, 720), (3840, 2160), (640, 360)])
def test_pixel_focal_from_physical_focal_and_actual_resolution(width, height):
    f_x, f_y = pixel_focal_lengths(35.0, width, height)
    assert f_x == pytest.approx(35.0 * width / 23.76)
    assert f_y == pytest.approx(35.0 * height / 13.365)
    cam = Camera.from_physical((0, 0, 0), 0, 0, 0, 35.0, width, height)
    np.testing.assert_allclose(cam.K, [[f_x, 0, width / 2], [0, f_y, height / 2], [0, 0, 1]])
    assert cam.focal_length_mm == 35.0                         # физический фокус не меняется
    assert cam.sensor_size_mm == (23.76, 13.365)               # матрица не меняется
    assert (cam.width, cam.height) == (width, height)


def test_sensor_size_is_constant():
    assert (SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM) == (23.76, 13.365)
    import pipeline.config as config

    assert config.SENSOR_SIZE_MM == (SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM)


def test_no_double_scaling(tmp_path):
    """make_camera для 1280×720 даёт ровно f·1280/23.76 — без повторного умножения на 1280/1920."""
    scene_csv(tmp_path, 1, [(0, 0, 0, 0, 0, 0, 0)], focal_mm=35)
    big, small = make_camera(tmp_path, 1, 0, (1920, 1080)), make_camera(tmp_path, 1, 0, (1280, 720))
    assert small.f_x == pytest.approx(35 * 1280 / 23.76) and small.f_y == pytest.approx(35 * 720 / 13.365)
    assert small.f_x / big.f_x == pytest.approx(1280 / 1920)
    assert big.focal_length_mm == small.focal_length_mm == 35.0
    # set_focal_length_mm пересчитывает от физического фокуса, а не от прежних f_x
    small.set_focal_length_mm(35.0)
    small.set_focal_length_mm(35.0)
    assert small.f_x == pytest.approx(35 * 1280 / 23.76)


def test_resolution_change_keeps_geometry():
    """Одна физическая камера в двух разрешениях: пиксели масштабируются, лучи и глубина — те же."""
    pts = np.array([[5000.0, 300.0, -200.0], [8000.0, -900.0, 400.0]])
    big = Camera.from_physical((0, 0, 100), 0.01, -0.05, 0.02, 35.0, 1920, 1080)
    small = Camera.from_physical((0, 0, 100), 0.01, -0.05, 0.02, 35.0, 960, 540)
    np.testing.assert_allclose(small.project(pts), big.project(pts) * 0.5)
    np.testing.assert_allclose(small.R, big.R)
    for cam in (big, small):
        in_camera = (pts - cam.position) @ cam.R.T
        rays = (np.linalg.inv(cam.K) @ np.c_[cam.project(pts), np.ones(2)].T).T     # K⁻¹ · (u, v, 1)
        np.testing.assert_allclose(rays[:, :2], in_camera[:, :2] / in_camera[:, 2:3])


def test_triangulation_with_cameras_of_different_resolutions():
    point = np.array([4000.0, 500.0, 200.0])

    def look_at(pos, f_mm, w, h):
        d = point - np.asarray(pos, float)
        return Camera.from_physical(pos, 0.0, math.atan2(d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0]),
                                    f_mm, w, h)

    cams = [look_at((0, -3000, 800), 35, 1920, 1080), look_at((0, 4000, 500), 50, 1280, 720),
            look_at((9000, 0, 1500), 25, 640, 360)]
    pixels = [c.project(point) for c in cams]
    np.testing.assert_allclose(triangulate(cams, pixels), point, atol=1e-3)


@pytest.mark.parametrize("width,height", [(1920, 1088), (640, 480), (1080, 1920)])
def test_aspect_ratio_other_than_sensor_is_rejected(width, height):
    """Кадр не 16:9 — обрезка или поля: формулы по всей матрице неприменимы."""
    with pytest.raises(ValueError, match="не соответствует матрице"):
        pixel_focal_lengths(35.0, width, height)


@pytest.mark.parametrize("width,height,focal", [(0, 1080, 35), (1920, -1, 35), (1920.5, 1080, 35),
                                                (1920, 1080, 0), (1920, 1080, float("nan"))])
def test_invalid_size_or_focal_is_rejected(width, height, focal):
    with pytest.raises(ValueError):
        Camera.from_physical((0, 0, 0), 0, 0, 0, focal, width, height)


# ---------------------------------------------------------------- разрешение видео

def test_video_resolution_from_file(tmp_path):
    assert probe_video_resolution(write_video(tmp_path / "a.mp4", 320, 180)) == (320, 180)
    assert probe_video_resolution(write_video(tmp_path / "b.mp4", 640, 360)) == (640, 360)


def test_each_camera_gets_its_own_video_resolution(tmp_path):
    sizes = {1: (320, 180), 2: (640, 360), 3: (480, 270)}
    for cam, (w, h) in sizes.items():
        write_video(tmp_path / f"Way2.FinalImage.CineCameraActor{cam}.mp4", w, h)
        scene_csv(tmp_path, cam, [(0, cam * 100, 0, 0, 0, 0, 0)], focal_mm=35)
    assert resolve_image_sizes(tmp_path, [1, 2, 3]) == sizes
    for cam, (w, h) in sizes.items():
        c = make_camera(tmp_path, cam, 0)                       # разрешение — из видео этой камеры
        assert (c.width, c.height) == (w, h) and c.f_x == pytest.approx(35 * w / 23.76)
        assert c.position[0] == cam * 100


def test_video_errors(tmp_path):
    with pytest.raises(VideoResolutionError, match="нет видео"):
        probe_video_resolution(tmp_path / "missing.mp4")
    (tmp_path / "broken.mp4").write_bytes(os.urandom(4096))
    with pytest.raises(VideoResolutionError):
        probe_video_resolution(tmp_path / "broken.mp4")
    empty = write_video(tmp_path / "empty.mp4", 320, 180, frames=0)
    with pytest.raises(VideoResolutionError):
        probe_video_resolution(empty)


def test_frame_resolution_checks_decoded_frame_against_metadata(caplog):
    img = np.zeros((360, 640, 3), np.uint8)
    with caplog.at_level(logging.WARNING, logger="peleng"):
        assert frame_resolution(img, "src", (1920.0, 1080.0)) == (640, 360)       # верен декодированный кадр
        assert "метаданные контейнера 1920×1080" in caplog.text
        assert frame_resolution(img, "src", (0.0, 0.0)) == (640, 360)             # метаданных нет
        assert "в метаданных нет разрешения" in caplog.text
    with pytest.raises(VideoResolutionError, match="не удалось декодировать"):
        frame_resolution(None, "src", (1920.0, 1080.0))
    with pytest.raises(VideoResolutionError):
        frame_resolution(np.zeros((0, 640, 3), np.uint8), "src")


def test_video_lookup_any_scene_prefix_but_not_ambiguous(tmp_path):
    (tmp_path / "Way2.FinalImage.CineCameraActor1.mp4").write_bytes(b"")
    (tmp_path / "Way2.FinalImage.CineCameraActor10.mp4").write_bytes(b"")
    assert find_video(tmp_path, 1).name == "Way2.FinalImage.CineCameraActor1.mp4"
    assert find_video(tmp_path, 2) is None
    (tmp_path / "Way1.FinalImage.CineCameraActor1.mp4").write_bytes(b"")
    with pytest.raises(VideoResolutionError, match="несколько видео"):
        find_video(tmp_path, 1)


def test_resolution_without_video_from_detections_meta(tmp_path):
    det = tmp_path / "detections"
    det.mkdir()
    (det / "meta.json").write_text(json.dumps({"weights": "w.pt"}), encoding="utf-8")
    write_detections_image_sizes(det, {1: (1280, 720)})
    write_detections_image_sizes(det, {2: (1920, 1080)})
    meta = json.loads((det / "meta.json").read_text(encoding="utf-8"))
    assert meta["weights"] == "w.pt" and meta["image_size"] == {"1": [1280, 720], "2": [1920, 1080]}
    assert detections_image_sizes(det) == {1: (1280, 720), 2: (1920, 1080)}
    assert resolve_image_sizes(tmp_path, [1, 2], det) == {1: (1280, 720), 2: (1920, 1080)}
    with pytest.raises(VideoResolutionError, match="cam3: разрешение кадра неизвестно"):
        resolve_image_sizes(tmp_path, [1, 3], det)
    with pytest.raises(VideoResolutionError, match="нет видео"):
        make_camera(tmp_path, 1, 0)                           # без видео и без переданного размера — ошибка


def test_video_and_detections_of_different_resolution_is_error(tmp_path):
    det = tmp_path / "detections"
    det.mkdir()
    write_detections_image_sizes(det, {1: (1920, 1080)})
    write_video(tmp_path / "Way2.FinalImage.CineCameraActor1.mp4", 640, 360)
    with pytest.raises(VideoResolutionError, match="пересчитайте детекции"):
        resolve_image_sizes(tmp_path, [1], det)


@requires_scene2
def test_real_scene2_videos_and_cameras():
    sizes = resolve_image_sizes(SCENE2_DIR, range(1, 10))
    assert sizes == {cam: (1920, 1080) for cam in range(1, 10)}
    cam = make_camera(SCENE2_DIR, 2, 0, sizes[2])
    assert cam.focal_length_mm == 60.0 and cam.f_x == pytest.approx(60 * 1920 / 23.76)


# ---------------------------------------------------------------- конфигурация

def test_config_has_no_fixed_resolution_and_legacy_keys(tmp_path, caplog):
    cfg = StreamConfig()
    assert not hasattr(cfg.cameras, "width") and not hasattr(cfg.cameras, "sensor_width_mm")
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"cameras": {"width": 1920, "height": 1080, "sensor_width_mm": 23.76,
                                           "sensor_height_mm": 13.365, "fps": 25.0}}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="peleng"):
        loaded = StreamConfig.load(old)
    assert loaded.cameras.fps == 25.0 and "cameras.width=1920 игнорируется" in caplog.text
    with pytest.raises(ValueError, match="постоянный"):
        StreamConfig.load(None, ["cameras.sensor_width_mm=36"])


# ---------------------------------------------------------------- интеграция: поток с разными разрешениями

NUM = 90
F_MM = 1500 * 23.76 / 1920            # фокус, при котором 1920×1080 даёт f_x = 1500 px (как в test_visibility)
CAM_POS = {1: (-3000, -7000, 1500), 2: (9000, -6000, 1800), 3: (5000, 8000, 1200)}
CAM_SIZE = {1: (1920, 1080), 2: (1280, 720), 3: (960, 540)}


def track_path(t):
    t = np.asarray(t, dtype=float)
    return np.c_[50 * t, 1500 * np.sin(t / 120), 300 + 0 * t]


def track_velocity(t):
    t = np.asarray(t, dtype=float)
    return np.c_[np.full_like(t, 50.0), 1500 / 120 * np.cos(t / 120), 0 * t]


def make_scene(data: Path) -> dict[int, dict[int, Detection]]:
    """CSV трёх камер (фокус в мм) и детекции, спроецированные камерами с их разрешениями."""
    target = track_path([45])[0]
    table = {}
    for cam, pos in CAM_POS.items():
        d = target - np.asarray(pos, float)
        pitch, yaw = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))), math.degrees(math.atan2(d[1], d[0]))
        scene_csv(data, cam, [(f, *pos, 0.0, pitch, yaw) for f in range(NUM)], focal_mm=F_MM)
        table[cam] = {}
        for f in range(NUM):
            camera = make_camera(data, cam, f, CAM_SIZE[cam])
            corners = box_corners(track_path([f]), track_velocity([f]), DEFAULT_DIMENSIONS.as_array())[0]
            uv = camera.project(corners)
            c = camera.project(corners.mean(axis=0))
            table[cam][f] = Detection(cam, f, 0.9, float(uv[:, 0].min()), float(uv[:, 1].min()),
                                      float(uv[:, 0].max()), float(uv[:, 1].max()), float(c[0]), float(c[1]),
                                      "mask_centroid", 5000.0, float(c[0]), float(c[1]))
    return table


def stream_config(mode="detections") -> StreamConfig:
    cfg = StreamConfig()
    cfg.cameras.ids = list(CAM_POS)
    cfg.streaming.mode = mode
    cfg.streaming.realtime_pacing = False
    cfg.streaming.log_every = 0
    cfg.spline.background = False
    cfg.tracker.edge_sigma_px = 2.0
    cfg.tracker.truck_fbx = ""                      # синтетика — коробка, FBX из models/ не подхватывать
    cfg.output.plots = False
    return cfg


def test_stream_detections_mode_with_different_resolutions(tmp_path):
    from pipeline.streaming import StreamPipeline

    data = tmp_path / "data"
    data.mkdir()
    table = make_scene(data)
    pipe = StreamPipeline(stream_config(), data, tmp_path / "out", table, NUM, image_sizes=CAM_SIZE)
    records = pipe.run()
    for cam, (w, h) in CAM_SIZE.items():                        # параметры дошли до камер конвейера
        camera = pipe.pose_provider(cam, 10)
        assert (camera.width, camera.height) == (w, h) and camera.f_x == pytest.approx(F_MM * w / 23.76)
        assert camera.focal_length_mm == pytest.approx(F_MM)
    pos = np.array([r.reference for r in records])
    err = np.linalg.norm(pos - track_path(np.arange(NUM)), axis=1)
    assert np.all(np.isfinite(pos[30:])) and np.nanmedian(err[40:]) < 60
    assert sum(r.status == "measured" for r in records) > NUM // 2


def test_stream_replay_reads_each_video_resolution(tmp_path):
    from pipeline.streaming import PrecomputedDetector, StreamPipeline

    data = tmp_path / "data"
    data.mkdir()
    table = make_scene(data)
    for cam, (w, h) in CAM_SIZE.items():
        write_video(data / f"Way2.FinalImage.CineCameraActor{cam}.mp4", w, h, frames=NUM)
    pipe = StreamPipeline(stream_config("replay"), data, tmp_path / "out", detector=PrecomputedDetector(table))
    records = pipe.run()
    assert pipe.image_sizes == CAM_SIZE                         # из файлов, а не из конфигурации
    assert len(records) == NUM
    pos = np.array([r.reference for r in records])
    err = np.linalg.norm(pos - track_path(np.arange(NUM)), axis=1)
    assert np.nanmedian(err[40:]) < 60


def test_stream_frame_size_change_rebuilds_intrinsics(tmp_path):
    from pipeline.streaming import StreamPipeline

    scene_csv(tmp_path, 1, [(0, 0, 0, 0, 0, 0, 0)], focal_mm=35)
    pipe = StreamPipeline(stream_config(), tmp_path, tmp_path / "out", {1: {}}, 1, image_sizes={1: (1920, 1080)})
    before = pipe._pose_from_csv(1, 0)
    pipe._observe_frame_size(1, np.zeros((720, 1280, 3), np.uint8))   # кадр потока другого размера
    after = pipe._pose_from_csv(1, 0)
    assert (after.width, after.height) == (1280, 720) and after.f_x == pytest.approx(35 * 1280 / 23.76)
    assert before.focal_length_mm == after.focal_length_mm == 35


def test_stream_camera_without_resolution_is_excluded_with_reason(tmp_path, caplog):
    from pipeline.streaming import Bundle, Frame, StreamPipeline

    scene_csv(tmp_path, 1, [(0, 0, 0, 0, 0, 0, 0)], focal_mm=35)
    scene_csv(tmp_path, 2, [(0, 0, 0, 0, 0, 0, 0)], focal_mm=35)
    pipe = StreamPipeline(stream_config(), tmp_path, tmp_path / "out", {1: {}, 2: {}}, 1,
                          image_sizes={1: (1920, 1080)})          # у камеры 2 нет ни видео, ни image_size
    bundle = Bundle(0, 0.0, {c: Frame(c, 0, 0.0, None, 0.0) for c in (1, 2)})
    with caplog.at_level(logging.ERROR, logger="peleng.stream"):
        poses = pipe._camera_poses(bundle)
    assert list(poses) == [1] and list(bundle.frames) == [1]
    assert "разрешение кадра неизвестно" in bundle.excluded[2] and "cam2" in caplog.text


# ---------------------------------------------------------------- экспорт координат

def test_trajectory_csv_matches_tracker_records(tmp_path):
    """trajectory.csv — те же координаты, что у трекера (центр кабины, м), без кадров инициализации."""
    import csv as _csv

    from pipeline.streaming import StreamPipeline

    data = tmp_path / "data"
    data.mkdir()
    table = make_scene(data)
    pipe = StreamPipeline(stream_config(), data, tmp_path / "out", table, NUM, image_sizes=CAM_SIZE)
    records = pipe.run()
    with open(tmp_path / "out" / "trajectory.csv", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert list(rows[0]) == ["frame_id", "timestamp", "object_id", "x", "y", "z", "camera_id", "status",
                             "position_std_m"]
    by_frame = {r.index: r for r in records}
    expected = [r for r in records if np.all(np.isfinite(r.cab_center))]
    assert len(rows) == len(expected) > NUM // 2
    for row in rows:
        r = by_frame[int(row["frame_id"])]
        np.testing.assert_allclose([float(row[k]) for k in "xyz"], r.cab_center / 100.0, atol=1e-4)
        assert row["status"] == r.status and row["object_id"] == "1"
        cams = [int(c) for c in row["camera_id"].split()]
        assert cams == ([] if r.status == "predicted" else list(r.detected_cameras))
