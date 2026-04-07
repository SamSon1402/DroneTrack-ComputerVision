"""Tests for pipeline utilities and helpers."""

import numpy as np
import pytest

from src.utils import bbox_center, bbox_area, iou, get_track_color, normalize_to_minimap


class TestUtils:
    def test_bbox_center(self):
        assert bbox_center([100, 200, 300, 400]) == (200, 300)
        assert bbox_center([0, 0, 640, 480]) == (320, 240)

    def test_bbox_area(self):
        assert bbox_area([0, 0, 100, 100]) == 10000
        assert bbox_area([0, 0, 0, 0]) == 0

    def test_iou_identical(self):
        box = [100, 100, 200, 200]
        assert abs(iou(box, box) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        a = [0, 0, 50, 50]
        b = [100, 100, 200, 200]
        assert iou(a, b) == 0.0

    def test_iou_partial(self):
        a = [0, 0, 100, 100]
        b = [50, 50, 150, 150]
        result = iou(a, b)
        assert 0 < result < 1.0

    def test_get_track_color(self):
        c1 = get_track_color(1)
        c2 = get_track_color(2)
        assert len(c1) == 3
        assert c1 != c2  # different IDs → different colors
        assert all(0 <= v <= 255 for v in c1)

    def test_normalize_to_minimap(self):
        mx, my = normalize_to_minimap(320, 240, 640, 480, 200)
        assert mx == 100
        assert my == 100


class TestExporter:
    def test_record_and_export(self, tmp_path):
        from src.exporter import TrackExporter
        from src.tracker import Track

        exporter = TrackExporter(output_dir=str(tmp_path))
        track = Track(
            track_id=1, bbox=[10, 20, 110, 120],
            class_name="Vehicle", confidence=0.92,
            age=5, time_since_update=0,
        )
        exporter.record(0, [track])
        exporter.record(1, [track])

        csv_path = exporter.export_csv("test.csv")
        json_path = exporter.export_json("test.json")

        assert (tmp_path / "test.csv").exists()
        assert (tmp_path / "test.json").exists()

        import json
        with open(json_path) as f:
            report = json.load(f)
        assert report["total_detections"] == 2
        assert report["unique_tracks"] == 1
