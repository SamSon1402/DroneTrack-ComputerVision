"""End-to-end detection → tracking → visualization pipeline."""

import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import yaml

from .detector import DroneDetector
from .tracker import DroneTracker
from .visualizer import Visualizer
from .hud import HUD
from .zone_alert import ZoneAlertSystem
from .exporter import TrackExporter


class DroneTrackPipeline:
    """Full pipeline: video → detect → track → visualize → export.

    Orchestrates all components and handles video I/O.
    """

    def __init__(self, config_path: str = "configs/default.yaml", **overrides):
        # Load config
        cfg = self._load_config(config_path)
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        self.cfg = cfg

        # Initialize components
        self.detector = DroneDetector(
            model_path=cfg.get("model", "yolov8n.pt"),
            conf_thresh=cfg.get("conf_thresh", 0.35),
            iou_thresh=cfg.get("iou_thresh", 0.45),
            input_size=cfg.get("input_size", 640),
            classes=cfg.get("classes"),
            class_map=cfg.get("class_map"),
            device=cfg.get("device", "auto"),
        )

        self.tracker = DroneTracker(
            max_age=cfg.get("max_age", 30),
            n_init=cfg.get("n_init", 3),
            max_iou_distance=cfg.get("max_iou_distance", 0.7),
            max_cosine_distance=cfg.get("max_cosine_distance", 0.3),
            nn_budget=cfg.get("nn_budget", 100),
            trail_length=cfg.get("trail_length", 40),
        )

        # Parse color config (convert lists to tuples)
        colors = cfg.get("colors")
        if colors:
            colors = {k: tuple(v) for k, v in colors.items()}

        self.visualizer = Visualizer(
            colors=colors,
            bbox_thickness=cfg.get("bbox_thickness", 2),
            font_scale=cfg.get("font_scale", 0.55),
            trail_thickness=cfg.get("trail_thickness", 2),
            show_trails=cfg.get("show_trails", True),
            show_minimap=cfg.get("show_minimap", True),
            minimap_size=cfg.get("minimap_size", 200),
            minimap_alpha=cfg.get("minimap_alpha", 0.7),
        )

        self.hud = HUD() if cfg.get("show_hud", True) else None
        self.zone_system = ZoneAlertSystem(cfg.get("zone_file"))
        self.exporter = TrackExporter(
            output_dir=str(Path(cfg.get("output", "output/tracked.mp4")).parent)
        )
        self.export_csv = cfg.get("export_csv", False)
        self.export_json = cfg.get("export_json", False)

    @staticmethod
    def _load_config(path: str) -> dict:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
        return {}

    def run(
        self,
        source: str,
        output: Optional[str] = None,
        display: bool = True,
    ) -> None:
        """Run the full pipeline on a video source.

        Args:
            source: Video file path, webcam index (as string "0"), or RTSP URL.
            output: Path to save annotated output video. None = no save.
            display: Whether to show live preview window.
        """
        # Open video source
        src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Pipeline] Source: {source} ({w}x{h} @ {fps_in:.0f}fps)")
        if total_frames > 0:
            print(f"[Pipeline] Total frames: {total_frames}")

        # Video writer
        writer = None
        out_path = output or self.cfg.get("output", "output/tracked.mp4")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))
        print(f"[Pipeline] Output: {out_path}")

        frame_idx = 0
        print("[Pipeline] Running... (press 'q' to stop)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()

                # 1. Detect
                detections = self.detector.detect(frame)

                # 2. Track
                tracks = self.tracker.update(detections, frame)

                inference_ms = (time.time() - t0) * 1000

                # 3. Zone alerts
                if self.zone_system.zones:
                    self.zone_system.draw_zones(frame)
                    intrusions = self.zone_system.check_intrusions(tracks)
                    for zone_name, track in intrusions:
                        print(
                            f"  ⚠️  ALERT: {track.class_name} #{track.track_id} "
                            f"entered '{zone_name}'"
                        )

                # 4. Visualize
                self.visualizer.draw(frame, tracks)
                if self.hud:
                    self.hud.draw(frame, tracks, inference_ms)

                # 5. Record for export
                self.exporter.record(frame_idx, tracks)

                # 6. Write output
                if writer:
                    writer.write(frame)

                # 7. Display
                if display:
                    cv2.imshow("DroneTrack-CV", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[Pipeline] Stopped by user.")
                        break

                frame_idx += 1

                # Progress
                if total_frames > 0 and frame_idx % 100 == 0:
                    pct = frame_idx / total_frames * 100
                    print(
                        f"  [{pct:5.1f}%] Frame {frame_idx}/{total_frames} "
                        f"| {len(tracks)} tracks | {inference_ms:.0f}ms"
                    )

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            print(f"\n[Pipeline] Processed {frame_idx} frames.")
            print(f"[Pipeline] Output saved → {out_path}")

            # Export
            if self.export_csv:
                self.exporter.export_csv()
            if self.export_json:
                self.exporter.export_json()

            print("[Pipeline] Done.")
