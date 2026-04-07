"""Tests for DroneTracker."""

import numpy as np
import pytest

from src.detector import Detection
from src.tracker import DroneTracker, Track


class TestDroneTracker:
    def test_init(self):
        tracker = DroneTracker()
        assert tracker.active_track_count == 0

    def test_update_empty(self):
        tracker = DroneTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = tracker.update([], frame)
        assert isinstance(tracks, list)

    def test_update_with_detections(self):
        tracker = DroneTracker(n_init=1)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection([100, 100, 200, 200], 0.9, 2, "Vehicle"),
            Detection([300, 300, 400, 400], 0.85, 0, "Person"),
        ]
        # Run multiple frames to confirm tracks
        for _ in range(4):
            tracks = tracker.update(detections, frame)
        # After n_init frames, tracks should be confirmed
        assert isinstance(tracks, list)

    def test_reset(self):
        tracker = DroneTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.update(
            [Detection([50, 50, 150, 150], 0.9, 2, "Vehicle")], frame
        )
        tracker.reset()
        assert tracker.active_track_count == 0


class TestTrack:
    def test_center(self):
        t = Track(
            track_id=1, bbox=[100, 200, 300, 400],
            class_name="Vehicle", confidence=0.9,
            age=5, time_since_update=0,
        )
        assert t.center == (200, 300)

    def test_is_confirmed(self):
        t = Track(
            track_id=1, bbox=[0, 0, 100, 100],
            class_name="Person", confidence=0.8,
            age=5, time_since_update=0,
        )
        assert t.is_confirmed is True

        t2 = Track(
            track_id=2, bbox=[0, 0, 100, 100],
            class_name="Person", confidence=0.8,
            age=1, time_since_update=0,
        )
        assert t2.is_confirmed is False
