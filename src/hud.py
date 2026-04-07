"""Telemetry HUD overlay for drone video output."""

import time
from typing import List
import cv2
import numpy as np

from .tracker import Track


class HUD:
    """Renders a heads-up display with FPS, object counts, and telemetry."""

    def __init__(self):
        self._frame_times: List[float] = []
        self._start_time: float = time.time()

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        inference_ms: float = 0.0,
    ) -> np.ndarray:
        """Draw HUD overlay on top of frame."""
        h, w = frame.shape[:2]
        now = time.time()

        # FPS calculation
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        fps = len(self._frame_times)

        # Elapsed time
        elapsed = now - self._start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        # Class counts
        class_counts: dict = {}
        for t in tracks:
            class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1

        # --- Top bar ---
        bar_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.45
        y_text = 22

        items = [
            (f"FPS: {fps}", (0, 220, 200)),
            (f"TRACKED: {len(tracks)}", (0, 180, 255)),
            (f"INF: {inference_ms:.0f}ms", (200, 200, 0)),
            (f"TIME: {minutes:02d}:{seconds:02d}", (180, 120, 255)),
        ]

        x = 12
        for text, color in items:
            cv2.putText(frame, text, (x, y_text), font, fs, color, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, font, fs, 1)
            x += tw + 24

        # Class counts on right side
        x_right = w - 12
        for cls_name, count in sorted(class_counts.items()):
            label = f"{cls_name}: {count}"
            (tw, _), _ = cv2.getTextSize(label, font, fs, 1)
            x_right -= tw + 20
            cv2.putText(
                frame, label, (x_right, y_text),
                font, fs, (180, 200, 220), 1, cv2.LINE_AA,
            )

        return frame

    def reset(self) -> None:
        self._frame_times.clear()
        self._start_time = time.time()
