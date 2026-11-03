import csv
import io

import pytest

from detection.schema import Detection


def test_detection_roundtrip_through_csv():
    d = Detection(camera=3, frame=17, confidence=0.91, x1=1.0, y1=2.0, x2=30.0, y2=40.0,
                  point_u=15.5, point_v=21.0, point_source="mask_centroid", mask_area_px=812.0,
                  keypoint_u=14.0, keypoint_v=39.0, edge_clipped=True, unreliable=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=Detection.columns())
    writer.writeheader()
    writer.writerow(d.to_row())
    buf.seek(0)
    restored = Detection.from_row(next(csv.DictReader(buf)))
    for name, value in d.to_row().items():
        got = getattr(restored, name)
        assert type(got) is type(value), name
        assert got == value or (value != value and got != got), name  # NaN == NaN


def test_detection_schema_fields():
    required = {"camera", "frame", "confidence", "x1", "y1", "x2", "y2", "point_u", "point_v",
                "mask_area_px", "keypoint_u", "keypoint_v", "edge_clipped", "too_small", "unreliable"}
    assert required <= set(Detection.columns())


def test_yolo_missing_weights_is_clear_error(tmp_path):
    from detection.yolo import WeightsNotFoundError, YoloDetector, YoloSettings

    with pytest.raises(WeightsNotFoundError, match="не найден"):
        YoloDetector(YoloSettings(weights_path=tmp_path / "best.pt"))
    bad = tmp_path / "best.pt-2"
    bad.write_bytes(b"x")
    with pytest.raises(WeightsNotFoundError, match=r"\.pt"):
        YoloDetector(YoloSettings(weights_path=bad))
