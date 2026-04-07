"""Export tracked objects to CSV and JSON formats."""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from .tracker import Track


class TrackExporter:
    """Accumulates track data and exports to CSV/JSON."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict] = []
        self._frame_idx: int = 0
        self._start_time: float = time.time()
        self._unique_ids: set = set()

    def record(self, frame_idx: int, tracks: List[Track]) -> None:
        """Record tracks for a single frame."""
        self._frame_idx = frame_idx
        timestamp = time.time() - self._start_time

        for track in tracks:
            self._unique_ids.add(track.track_id)
            self._records.append({
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "track_id": track.track_id,
                "class": track.class_name,
                "confidence": round(track.confidence, 3),
                "x1": int(track.bbox[0]),
                "y1": int(track.bbox[1]),
                "x2": int(track.bbox[2]),
                "y2": int(track.bbox[3]),
                "cx": track.center[0],
                "cy": track.center[1],
            })

    def export_csv(self, filename: str = "tracks.csv") -> str:
        """Export all records to CSV."""
        path = self.output_dir / filename
        if not self._records:
            return str(path)

        fieldnames = list(self._records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)

        print(f"[Exporter] CSV saved → {path} ({len(self._records)} rows)")
        return str(path)

    def export_json(self, filename: str = "report.json") -> str:
        """Export summary report as JSON."""
        path = self.output_dir / filename

        # Compute class distribution
        class_counts: Dict[str, int] = {}
        for r in self._records:
            cls = r["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        report = {
            "pipeline": "DroneTrack-CV",
            "version": "1.0.0",
            "total_frames": self._frame_idx + 1,
            "total_detections": len(self._records),
            "unique_tracks": len(self._unique_ids),
            "class_distribution": class_counts,
            "duration_seconds": round(time.time() - self._start_time, 2),
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[Exporter] JSON report saved → {path}")
        return str(path)

    def reset(self) -> None:
        self._records.clear()
        self._frame_idx = 0
        self._unique_ids.clear()
        self._start_time = time.time()
