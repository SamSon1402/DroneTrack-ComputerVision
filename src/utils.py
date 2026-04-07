"""Utility functions for DroneTrack-CV."""

from typing import Dict, List, Tuple
import colorsys
import numpy as np


# Default BGR colors per class
DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Vehicle": (212, 182, 6),    # Cyan
    "Person": (22, 115, 249),    # Orange
    "Cyclist": (8, 171, 234),    # Yellow
    "Anomaly": (68, 68, 239),    # Red
}

# COCO class IDs → DroneTrack class names
COCO_CLASS_MAP: Dict[int, str] = {
    0: "Person",
    1: "Cyclist",
    2: "Vehicle",   # car
    3: "Vehicle",   # motorcycle
    5: "Vehicle",   # bus
    7: "Vehicle",   # truck
}


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a unique, visually distinct BGR color for a track ID."""
    hue = (track_id * 0.618033988749895) % 1.0  # golden ratio for spread
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def bbox_center(bbox: List[float]) -> Tuple[int, int]:
    """Return (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))


def bbox_area(bbox: List[float]) -> float:
    """Return area of a bounding box [x1, y1, x2, y2]."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two bounding boxes [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_to_minimap(
    cx: int, cy: int, frame_w: int, frame_h: int, map_size: int
) -> Tuple[int, int]:
    """Map frame coordinates to minimap coordinates."""
    mx = int((cx / frame_w) * map_size)
    my = int((cy / frame_h) * map_size)
    return (mx, my)
