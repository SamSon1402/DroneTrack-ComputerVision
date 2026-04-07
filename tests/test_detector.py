"""Tests for DroneDetector."""

import numpy as np
import pytest


class TestDetection:
    """Test Detection data class."""

    def test_to_ltwh(self):
        from src.detector import Detection

        det = Detection(
            bbox=[100, 200, 300, 400],
            confidence=0.9,
            class_id=2,
            class_name="Vehicle",
        )
        ltwh = det.to_ltwh()
        assert ltwh == [100, 200, 200, 200]

    def test_repr(self):
        from src.detector import Detection

        det = Detection(
            bbox=[10.5, 20.3, 50.7, 80.1],
            confidence=0.85,
            class_id=0,
            class_name="Person",
        )
        r = repr(det)
        assert "Person" in r
        assert "0.85" in r


class TestDroneDetector:
    """Test DroneDetector initialization and detection."""

    def test_init_default(self):
        """Test detector initializes with default YOLOv8n model."""
        from src.detector import DroneDetector

        detector = DroneDetector(device="cpu")
        assert detector.conf_thresh == 0.35
        assert detector.input_size == 640

    def test_detect_returns_list(self):
        """Test detection on a blank frame returns empty or valid list."""
        from src.detector import DroneDetector

        detector = DroneDetector(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)

    def test_detect_on_random_frame(self):
        """Test detection on noise frame doesn't crash."""
        from src.detector import DroneDetector

        detector = DroneDetector(device="cpu")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)
        for det in detections:
            assert det.class_name in ("Vehicle", "Person", "Cyclist")
            assert 0 < det.confidence <= 1.0
            assert len(det.bbox) == 4
