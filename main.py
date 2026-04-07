#!/usr/bin/env python3
"""DroneTrack-CV — Real-Time Aerial Multi-Object Tracking.

CLI entry point. Run `python main.py --help` for usage.

Example:
    python main.py --source drone_video.mp4 --show-trails --show-minimap --show-hud
"""

import argparse
import sys

from src.pipeline import DroneTrackPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DroneTrack-CV: Real-Time Aerial Multi-Object Detection & Tracking",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Input / Output
    parser.add_argument(
        "--source", required=True,
        help="Video file path, webcam index (0), or RTSP URL",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output video path (default: output/tracked.mp4)",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Config YAML path (default: configs/default.yaml)",
    )

    # Detection
    parser.add_argument("--model", default=None, help="YOLOv8 model path (e.g. yolov8n.pt)")
    parser.add_argument("--conf-thresh", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=None, help="NMS IoU threshold")

    # Tracking
    parser.add_argument("--track-thresh", type=float, default=None, help="DeepSORT track threshold")
    parser.add_argument("--max-age", type=int, default=None, help="Max frames to keep lost track")

    # Visualization
    parser.add_argument("--show-trails", action="store_true", help="Draw trajectory trails")
    parser.add_argument("--no-trails", action="store_true", help="Disable trajectory trails")
    parser.add_argument("--trail-length", type=int, default=None, help="Max trail points per track")
    parser.add_argument("--show-minimap", action="store_true", help="Show bird's-eye minimap")
    parser.add_argument("--no-minimap", action="store_true", help="Disable minimap")
    parser.add_argument("--show-hud", action="store_true", help="Show telemetry HUD")
    parser.add_argument("--no-hud", action="store_true", help="Disable HUD")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no preview window)")

    # Zone alerts
    parser.add_argument("--zone-file", default=None, help="JSON file defining restricted zones")

    # Export
    parser.add_argument("--export-csv", action="store_true", help="Export tracks to CSV")
    parser.add_argument("--export-json", action="store_true", help="Export summary to JSON")

    # Device
    parser.add_argument("--device", default=None, help="Device: cuda:0, cpu, or auto")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build override dict from CLI args
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.conf_thresh is not None:
        overrides["conf_thresh"] = args.conf_thresh
    if args.iou_thresh is not None:
        overrides["iou_thresh"] = args.iou_thresh
    if args.track_thresh is not None:
        overrides["track_thresh"] = args.track_thresh
    if args.max_age is not None:
        overrides["max_age"] = args.max_age
    if args.trail_length is not None:
        overrides["trail_length"] = args.trail_length
    if args.device:
        overrides["device"] = args.device
    if args.zone_file:
        overrides["zone_file"] = args.zone_file
    if args.output:
        overrides["output"] = args.output

    # Boolean flags
    if args.show_trails:
        overrides["show_trails"] = True
    if args.no_trails:
        overrides["show_trails"] = False
    if args.show_minimap:
        overrides["show_minimap"] = True
    if args.no_minimap:
        overrides["show_minimap"] = False
    if args.show_hud:
        overrides["show_hud"] = True
    if args.no_hud:
        overrides["show_hud"] = False
    if args.export_csv:
        overrides["export_csv"] = True
    if args.export_json:
        overrides["export_json"] = True

    print("=" * 60)
    print("  🛸 DroneTrack-CV — Real-Time Aerial Object Tracking")
    print("     Sky-Drones Technologies")
    print("=" * 60)
    print()

    # Initialize and run
    pipeline = DroneTrackPipeline(
        config_path=args.config,
        **overrides,
    )

    pipeline.run(
        source=args.source,
        output=args.output,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
