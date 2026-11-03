import numpy as np
import pytest

from reconstruction.cab_model import TruckDimensions, box_corners
from reconstruction.fbx_reader import load_meshes, read_fbx
from reconstruction.objective import ObservationGeometry
from reconstruction.truck_model import TruckModel
from reconstruction.visibility import CabTracker, TrackerSettings, run_tracker
from tests.conftest import REPO_ROOT
from tests.test_visibility import CAMERAS, FRAMES, errors, provider, synthetic

FBX = REPO_ROOT / "models" / "CargoTruck.fbx"
requires_fbx = pytest.mark.skipif(not FBX.exists(), reason=f"нет 3D-модели {FBX}")


@requires_fbx
def test_fbx_structure_and_units():
    root = read_fbx(FBX)
    assert root.properties[0] == 7400
    scene = load_meshes(FBX)
    names = {m.name for m in scene.meshes}
    assert {"Truck", "Cargo", "FrontTires"} <= names
    vertices = np.vstack([m.vertices for m in scene.meshes])
    size = vertices.max(0) - vertices.min(0)
    np.testing.assert_allclose(size, [167.8, 242.4, 668.5], atol=0.5)        # X, Y (вверх), Z (вперёд), см
    assert vertices[:, 1].min() == pytest.approx(-2.2, abs=0.5)              # колёса стоят на земле
    front = [m for m in scene.meshes if m.name == "FrontTires"][0]
    assert front.vertices[:, 2].mean() > 0                                    # кабина на +Z


@requires_fbx
def test_truck_model_points_and_scale():
    model = TruckModel.from_fbx(FBX, scale=1.0)
    np.testing.assert_allclose(model.dimensions, [668.5, 167.8, 242.4], atol=0.5)
    bigger = TruckModel.from_fbx(FBX, scale=3.0)
    np.testing.assert_allclose(bigger.dimensions, 3 * model.dimensions, rtol=1e-9)
    half = model.dimensions / 2
    assert np.all(np.abs(model.hull) <= half + 1e-6)
    # Центр кабины — впереди и выше середины, внутри габаритов; точка CSV — у переднего края
    assert 0 < model.cab_center[0] < half[0] and model.cab_center[2] > -half[2]
    assert model.csv_point[0] > model.cab_center[0]
    assert model.front_bottom[0] == pytest.approx(half[0]) and model.front_bottom[2] == pytest.approx(-half[2])


def box_model(dims: TruckDimensions) -> TruckModel:
    """Модель-коробка: должна давать те же измерения, что и трекер с коробкой."""
    corners = box_corners(np.zeros((1, 3)), np.array([[1.0, 0, 0]]), dims.as_array())[0]
    center = (corners.min(0) + corners.max(0)) / 2
    d = dims.as_array()
    return TruckModel(hull=corners - center, dimensions=d, csv_point=np.array([d[0] / 2, 0, -d[2] / 2]),
                      cab_center=np.zeros(3), front_bottom=np.array([d[0] / 2, 0, -d[2] / 2]))


def test_model_measurement_equals_box_for_box_shape():
    dims = TruckDimensions(1650.0, 255.0, 400.0)
    box_tracker = CabTracker(dimensions=dims)
    model_tracker = CabTracker(model=box_model(dims))
    geometry = ObservationGeometry.from_cameras(CAMERAS)
    center = np.array([4000.0, 500.0, 500.0])
    direction = np.array([0.9, 0.3, 0.05])
    np.testing.assert_allclose(model_tracker._measure(center, direction, geometry),
                               box_tracker._measure(center, direction, geometry), atol=1e-6)


def test_tracker_with_box_model_matches_box_tracker():
    dims = TruckDimensions(1650.0, 255.0, 400.0)
    obs, points = synthetic(seed=7)
    settings = TrackerSettings(edge_sigma_px=2.0)
    a = run_tracker(obs, provider, FRAMES, points, settings=settings, dimensions=dims)
    b = run_tracker(obs, provider, FRAMES, points, settings=settings, model=box_model(dims))
    np.testing.assert_allclose(a.array("box_center")[20:], b.array("box_center")[20:], atol=1e-3)
    err, _ = errors(b)
    assert np.median(err[40:]) < 50
