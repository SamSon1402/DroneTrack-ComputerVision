"""DeepSORT multi-object tracker wrapper."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from .detector import Detection
from .utils import bbox_center


@dataclass
class Track:
    """Tracked object with persistent identity."""

    track_id: int
    bbox: List[float]          # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    age: int                   # frames since first seen
    time_since_update: int     # frames since last detection match
    trajectory: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[int, int]:
        return bbox_center(self.bbox)

    @property
    def is_confirmed(self) -> bool:
        return self.age >= 3 and self.time_since_update == 0


class DroneTracker:
    """DeepSORT-based multi-object tracker.

    Maintains persistent track IDs across frames, handles occlusions
    via Kalman prediction + Re-ID appearance matching.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100,
        trail_length: int = 40,
    ):
        self.deepsort = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )
        self.trail_length = trail_length
        self._trajectories: Dict[int, List[Tuple[int, int]]] = {}
        self._class_memory: Dict[int, str] = {}
        self._conf_memory: Dict[int, float] = {}

    def update(
        self, detections: List[Detection], frame: np.ndarray
    ) -> List[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of Detection objects from detector.
            frame: Current BGR frame (used for Re-ID feature extraction).

        Returns:
            List of confirmed Track objects with updated trajectories.
        """
        if not detections:
            raw_tracks = self.deepsort.update_tracks([], frame=frame)
        else:
            # Format detections for deep_sort_realtime:
            #   list of ([left, top, w, h], confidence, class_name)
            ds_detections = []
            for det in detections:
                ltwh = det.to_ltwh()
                ds_detections.append((ltwh, det.confidence, det.class_name))

            raw_tracks = self.deepsort.update_tracks(
                ds_detections, frame=frame
            )

        # Build Track objects
        tracks: List[Track] = []
        for rt in raw_tracks:
            if not rt.is_confirmed():
                continue

            track_id = rt.track_id
            ltrb = rt.to_ltrb()  # [left, top, right, bottom]
            bbox = [float(v) for v in ltrb]
            det_class = rt.det_class if rt.det_class else "Unknown"

            # Persist class and confidence across frames
            if det_class != "Unknown":
                self._class_memory[track_id] = det_class
            class_name = self._class_memory.get(track_id, det_class)

            conf = rt.det_conf if rt.det_conf else 0.0
            if conf > 0:
                self._conf_memory[track_id] = conf
            confidence = self._conf_memory.get(track_id, conf)

            # Update trajectory
            center = bbox_center(bbox)
            if track_id not in self._trajectories:
                self._trajectories[track_id] = []
            self._trajectories[track_id].append(center)
            if len(self._trajectories[track_id]) > self.trail_length:
                self._trajectories[track_id] = self._trajectories[track_id][
                    -self.trail_length :
                ]

            tracks.append(
                Track(
                    track_id=track_id,
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence,
                    age=rt.age,
                    time_since_update=rt.time_since_update,
                    trajectory=list(self._trajectories[track_id]),
                )
            )

        return tracks

    def reset(self) -> None:
        """Reset tracker state."""
        self.deepsort = DeepSort(
            max_age=self.deepsort.max_age,
            n_init=self.deepsort.n_init,
        )
        self._trajectories.clear()
        self._class_memory.clear()
        self._conf_memory.clear()

    @property
    def active_track_count(self) -> int:
        return len(self._trajectories)
