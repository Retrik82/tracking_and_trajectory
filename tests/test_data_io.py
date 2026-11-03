import numpy as np

from reconstruction.data_io import camera_csv_path, load_target, read_focal_length_mm
from tests.conftest import DATA_DIR, FOCAL_MM, requires_data


@requires_data
def test_focal_length_from_header():
    for cam, focal in FOCAL_MM.items():
        assert read_focal_length_mm(camera_csv_path(DATA_DIR, cam)) == focal


@requires_data
def test_target_corrupted_cells_are_reported_not_guessed():
    target = load_target(DATA_DIR)
    assert len(target.frames) == 1002
    assert len(target.issues) == 2
    assert np.isnan(target.position(402)[0]) and np.isfinite(target.position(402)[1:]).all()
    assert np.isnan(target.position(691)[1]) and np.isfinite(target.position(691)[[0, 2]]).all()
    assert target.valid.sum() == 1000
    assert np.isnan(target.position(5000)).all()


def test_target_reader_on_synthetic_file(tmp_path):
    content = "Frame;Loc_X;Loc_Y;Loc_Z\r\n0;1.5;2,5;3\r\n1;авг.13;2;3\r\n2;-1e2;;3\r\n"
    (tmp_path / "object_transform_SM_AA_track.csv").write_bytes(content.encode("cp1251"))
    target = load_target(tmp_path)
    np.testing.assert_allclose(target.position(0), [1.5, 2.5, 3])
    assert np.isnan(target.position(1)[0])
    assert np.isnan(target.position(2)[1]) and target.position(2)[0] == -100
    assert len(target.issues) == 2
