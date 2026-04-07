"""Zone-based intrusion detection for restricted areas."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from .tracker import Track
from .utils import bbox_center


class Zone:
    """A polygonal restricted zone."""

    def __init__(self, name: str, points: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 0, 255)):
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.color = color

    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside this zone."""
        return cv2.pointPolygonTest(self.points, point, False) >= 0


class ZoneAlertSystem:
    """Monitors tracks for zone intrusions and draws zone overlays."""

    def __init__(self, zone_file: Optional[str] = None):
        self.zones: List[Zone] = []
        self._alerted_tracks: Dict[str, Set[int]] = {}  # zone_name → set of track IDs already alerted

        if zone_file and Path(zone_file).exists():
            self._load_zones(zone_file)

    def _load_zones(self, path: str) -> None:
        """Load zones from a JSON file.

        Expected format:
        [
            {"name": "Restricted A", "points": [[100,100],[300,100],[300,400],[100,400]], "color": [0,0,255]},
            ...
        ]
        """
        with open(path) as f:
            data = json.load(f)
        for z in data:
            self.zones.append(
                Zone(
                    name=z["name"],
                    points=[tuple(p) for p in z["points"]],
                    color=tuple(z.get("color", [0, 0, 255])),
                )
            )
            self._alerted_tracks[z["name"]] = set()

    def check_intrusions(self, tracks: List[Track]) -> List[Tuple[str, Track]]:
        """Check which tracks are inside restricted zones.

        Returns:
            List of (zone_name, track) tuples for new intrusions.
        """
        intrusions: List[Tuple[str, Track]] = []
        for zone in self.zones:
            for track in tracks:
                if zone.contains(track.center):
                    if track.track_id not in self._alerted_tracks[zone.name]:
                        self._alerted_tracks[zone.name].add(track.track_id)
                        intrusions.append((zone.name, track))
        return intrusions

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw zone overlays onto frame."""
        overlay = frame.copy()
        for zone in self.zones:
            cv2.fillPoly(overlay, [zone.points], zone.color)
            cv2.polylines(frame, [zone.points], True, zone.color, 2)

            # Zone label
            cx = int(zone.points[:, 0].mean())
            cy = int(zone.points[:, 1].mean())
            cv2.putText(
                frame, zone.name, (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        return frame

    def reset(self) -> None:
        for key in self._alerted_tracks:
            self._alerted_tracks[key].clear()
