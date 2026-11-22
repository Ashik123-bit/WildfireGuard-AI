from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.detection.detector import Detection
from src.prediction.spread_predictor import bounding_box_coverage


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    fire_coverage: float
    smoke_detected: bool
    maximum_confidence: float
    note: str
    recommendation: str


def assess_detection_risk(
    detections: Iterable[Detection], image_width: int, image_height: int
) -> RiskAssessment:
    detections = tuple(detections)
    fire_coverage = bounding_box_coverage(
        detections, image_width, image_height, label="fire"
    )
    smoke_detected = any(item.label == "smoke" for item in detections)
    maximum_confidence = max((item.confidence for item in detections), default=0.0)

    if detections and maximum_confidence < 0.35:
        level = "Uncertain"
        note = "Low-confidence detections need human verification."
        recommendation = "Require human verification."
    elif fire_coverage >= 0.20:
        level = "High"
        note = "Fire boxes cover a large share of this image; verify immediately."
        recommendation = "Promptly notify relevant local emergency authorities."
    elif fire_coverage >= 0.05 or smoke_detected:
        level = "Elevated"
        note = "Fire or smoke was detected; verify the scene and local conditions."
        recommendation = "Increase monitoring and seek human verification."
    elif detections:
        level = "Detected"
        note = "A small visible fire or smoke region was detected."
        recommendation = "Verify the detection."
    else:
        level = "No detection"
        note = "No fire or smoke was detected in the analyzed frame."
        recommendation = "Continue routine monitoring."

    return RiskAssessment(
        level=level,
        fire_coverage=fire_coverage,
        smoke_detected=smoke_detected,
        maximum_confidence=maximum_confidence,
        note=note,
        recommendation=recommendation,
    )
