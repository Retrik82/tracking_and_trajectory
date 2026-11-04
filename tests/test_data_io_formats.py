"""Форматы данных двух сцен: фокус (заголовок / колонка), разрешение из файла, поиск видео."""

import numpy as np
import pytest

from reconstruction.data_io import (IMAGE_HEIGHT, IMAGE_WIDTH, read_focal_length_mm,
                                    video_path, video_resolution)


def _write(path, text):
    path.write_text(text, encoding="cp1251")
    return path


def test_focal_from_header_pair(tmp_path):
    """Сцена 1: «...;;;Focal length;35» в заголовке."""
    p = _write(tmp_path / "cam.csv",
               "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35\n"
               "0;1;2;3;0;0;0\n")
    assert read_focal_length_mm(p) == pytest.approx(35.0)


def test_focal_from_column(tmp_path):
    """Сцена 2: колонка Focal_Length_mm, значение в строках данных."""
    p = _write(tmp_path / "cam.csv",
               "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal_Length_mm\n"
               "0;1;2;3;0;0;0;25.0\n"
               "1;1;2;3;0;0;0;25.0\n")
    assert read_focal_length_mm(p) == pytest.approx(25.0)


def test_focal_missing_is_clear_error(tmp_path):
    p = _write(tmp_path / "cam.csv", "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw\n0;1;2;3;0;0;0\n")
    with pytest.raises(ValueError, match="фокусного расстояния"):
        read_focal_length_mm(p)


def test_focal_comma_decimal(tmp_path):
    p = _write(tmp_path / "cam.csv",
               "Frame;X;Y;Z;R;P;Yaw;Focal_Length_mm\n0;1;2;3;0;0;0;23,5\n")
    assert read_focal_length_mm(p) == pytest.approx(23.5)


def test_video_resolution_fallback_without_file(tmp_path):
    assert video_resolution(tmp_path, 1) == (IMAGE_WIDTH, IMAGE_HEIGHT)


def test_video_path_any_scene_prefix(tmp_path):
    (tmp_path / "Way2.FinalImage.CineCameraActor3.mp4").write_bytes(b"")
    assert video_path(tmp_path, 3).name == "Way2.FinalImage.CineCameraActor3.mp4"
    # нет файла — имя сцены 1 (для понятных сообщений об ошибке)
    assert video_path(tmp_path, 4).name == "Way1.FinalImage.CineCameraActor4.mp4"


def test_make_camera_uses_video_resolution_fallback(tmp_path):
    """Без видео фокусные расстояния считаются от разрешения по умолчанию."""
    from reconstruction.data_io import SENSOR_WIDTH_MM, make_camera

    _write(tmp_path / "object_transform_CineCameraActor1.csv",
           "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal_Length_mm\n"
           "0;0;0;100;0;0;0;25.0\n")
    cam = make_camera(tmp_path, 1, 0)
    assert cam.K[0, 0] == pytest.approx(25.0 / SENSOR_WIDTH_MM * IMAGE_WIDTH)
    assert np.allclose(cam.K[:2, 2], [IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2])
