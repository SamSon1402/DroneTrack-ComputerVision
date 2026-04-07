"""YOLOv8 detection wrapper for aerial/drone imagery."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from .utils import COCO_CLASS_MAP


class Detection:
    """Single detection result."""

    __slots__ = ("bbox", "confidence", "class_id", "class_name")

    def __init__(
        self,
        bbox: List[float],
        confidence: float,
        class_id: int,
        class_name: str,
    ):
        self.bbox = bbox              # [x1, y1, x2, y2] absolute pixels
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def to_ltwh(self) -> List[float]:
        """Convert to [left, top, width, height] for DeepSORT."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={[int(v) for v in self.bbox]})"
        )


class DroneDetector:
    """YOLOv8-based object detector tuned for aerial footage.

    Wraps ultralytics YOLO and filters to drone-relevant COCO classes
    (person, car, truck, bus, motorcycle, bicycle).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        input_size: int = 640,
        classes: Optional[List[int]] = None,
        class_map: Optional[Dict[int, str]] = None,
        device: str = "auto",
    ):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.classes = classes or list(COCO_CLASS_MAP.keys())
        self.class_map = class_map or COCO_CLASS_MAP
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Returns:
            List of Detection objects for drone-relevant classes.
        """
        results = self.model.predict(
            source=frame,
            imgsz=self.input_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
            class_name = self.class_map.get(cls_id, "Unknown")
            if class_name == "Unknown":
                continue
            detections.append(
                Detection(
                    bbox=bbox.tolist(),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name,
                )
            )

        return detections

    def export_onnx(self, output_path: str = "dronetrack.onnx") -> str:
        """Export model to ONNX format for edge deployment."""
        self.model.export(format="onnx", imgsz=self.input_size)
        print(f"[DroneDetector] ONNX model exported → {output_path}")
        return output_path
