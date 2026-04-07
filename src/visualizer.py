"""Visualization engine: bounding boxes, trajectory trails, and spatial minimap."""

from typing import Dict, List, Tuple
import cv2
import numpy as np

from .tracker import Track
from .utils import DEFAULT_COLORS, get_track_color, normalize_to_minimap


class Visualizer:
    """Renders detection/tracking overlays onto video frames."""

    def __init__(
        self,
        colors: Dict[str, Tuple[int, int, int]] = None,
        bbox_thickness: int = 2,
        font_scale: float = 0.55,
        trail_thickness: int = 2,
        show_trails: bool = True,
        show_minimap: bool = True,
        minimap_size: int = 200,
        minimap_alpha: float = 0.7,
    ):
        self.colors = colors or DEFAULT_COLORS
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.trail_thickness = trail_thickness
        self.show_trails = show_trails
        self.show_minimap = show_minimap
        self.minimap_size = minimap_size
        self.minimap_alpha = minimap_alpha

    def draw(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw all overlays onto a frame (in-place).

        Args:
            frame: BGR image (will be modified in-place).
            tracks: List of confirmed Track objects.

        Returns:
            The annotated frame.
        """
        for track in tracks:
            color = self.colors.get(track.class_name, (200, 200, 200))
            self._draw_bbox(frame, track, color)
            if self.show_trails and len(track.trajectory) > 1:
                self._draw_trail(frame, track)

        if self.show_minimap:
            self._draw_minimap(frame, tracks)

        return frame

    def _draw_bbox(
        self, frame: np.ndarray, track: Track, color: Tuple[int, int, int]
    ) -> None:
        """Draw bounding box with label."""
        x1, y1, x2, y2 = [int(v) for v in track.bbox]

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)

        # Corner accents (tactical style)
        corner_len = min(20, int((x2 - x1) * 0.3))
        t = self.bbox_thickness + 1
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

        # Label background
        label = f"#{track.track_id} {track.class_name} {track.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )
        label_y = max(y1 - 8, th + 6)
        cv2.rectangle(
            frame,
            (x1, label_y - th - 6),
            (x1 + tw + 8, label_y + 2),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            1,
        )

    def _draw_trail(self, frame: np.ndarray, track: Track) -> None:
        """Draw fading trajectory trail."""
        pts = track.trajectory
        n = len(pts)
        color = get_track_color(track.track_id)

        for i in range(1, n):
            alpha = i / n  # fade: older points are more transparent
            thickness = max(1, int(self.trail_thickness * alpha))
            pt_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts[i - 1], pts[i], pt_color, thickness)

        # Current position dot
        if pts:
            cv2.circle(frame, pts[-1], 4, color, -1)

    def _draw_minimap(
        self, frame: np.ndarray, tracks: List[Track]
    ) -> None:
        """Draw bird's-eye spatial minimap in bottom-right corner."""
        h, w = frame.shape[:2]
        ms = self.minimap_size

        # Create minimap background
        minimap = np.zeros((ms, ms, 3), dtype=np.uint8)

        # Grid lines
        for i in range(0, ms, ms // 5):
            cv2.line(minimap, (i, 0), (i, ms), (30, 40, 50), 1)
            cv2.line(minimap, (0, i), (ms, i), (30, 40, 50), 1)

        # Border
        cv2.rectangle(minimap, (0, 0), (ms - 1, ms - 1), (50, 60, 80), 1)

        # Drone icon (center)
        center = (ms // 2, ms // 2)
        cv2.circle(minimap, center, 5, (180, 100, 255), -1)
        cv2.circle(minimap, center, 12, (180, 100, 255), 1)

        # Plot tracked objects
        for track in tracks:
            cx, cy = track.center
            mx, my = normalize_to_minimap(cx, cy, w, h, ms)
            color = self.colors.get(track.class_name, (200, 200, 200))
            cv2.circle(minimap, (mx, my), 3, color, -1)

            # Mini trail
            if len(track.trajectory) > 2:
                trail_pts = track.trajectory[-10:]
                for pt in trail_pts:
                    tx, ty = normalize_to_minimap(pt[0], pt[1], w, h, ms)
                    cv2.circle(minimap, (tx, ty), 1, color, -1)

        # Label
        cv2.putText(
            minimap, "SPATIAL MAP", (6, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 120, 150), 1,
        )

        # Blend onto frame
        roi_x = w - ms - 15
        roi_y = h - ms - 15
        roi = frame[roi_y : roi_y + ms, roi_x : roi_x + ms]
        blended = cv2.addWeighted(roi, 1 - self.minimap_alpha, minimap, self.minimap_alpha, 0)
        frame[roi_y : roi_y + ms, roi_x : roi_x + ms] = blended
