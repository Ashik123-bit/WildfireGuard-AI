from __future__ import annotations

import math
from collections.abc import Iterable

from src.detection.detector import Detection


def bounding_box_coverage(
    detections: Iterable[Detection],
    image_width: int,
    image_height: int,
    label: str = "fire",
) -> float:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    rectangles: list[tuple[float, float, float, float]] = []
    for detection in detections:
        if detection.label != label:
            continue
        x1 = min(max(detection.x1, 0.0), float(image_width))
        y1 = min(max(detection.y1, 0.0), float(image_height))
        x2 = min(max(detection.x2, 0.0), float(image_width))
        y2 = min(max(detection.y2, 0.0), float(image_height))
        if x2 > x1 and y2 > y1:
            rectangles.append((x1, y1, x2, y2))

    x_edges = sorted({edge for rect in rectangles for edge in (rect[0], rect[2])})
    covered_area = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (rect[1], rect[3])
            for rect in rectangles
            if rect[0] < right and rect[2] > left
        )
        covered_height = 0.0
        if intervals:
            interval_start, interval_end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start <= interval_end:
                    interval_end = max(interval_end, next_end)
                else:
                    covered_height += interval_end - interval_start
                    interval_start, interval_end = next_start, next_end
            covered_height += interval_end - interval_start
        covered_area += (right - left) * covered_height

    return covered_area / (image_width * image_height)


def estimate_visible_area_hectares(
    coverage_fraction: float, camera_footprint_hectares: float
) -> float:
    if not math.isfinite(coverage_fraction) or not 0 <= coverage_fraction <= 1:
        raise ValueError("coverage_fraction must be between zero and one.")
    if not math.isfinite(camera_footprint_hectares) or camera_footprint_hectares < 0:
        raise ValueError("camera_footprint_hectares must be a non-negative number.")
    return coverage_fraction * camera_footprint_hectares


def project_spread_hectares(
    initial_area_hectares: float,
    radial_spread_metres_per_hour: float,
    hours: float,
) -> float:
    values = (initial_area_hectares, radial_spread_metres_per_hour, hours)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Spread inputs must be finite numbers.")
    if any(value < 0 for value in values):
        raise ValueError("Spread inputs must be non-negative.")

    initial_area_square_metres = initial_area_hectares * 10_000
    initial_radius_metres = math.sqrt(initial_area_square_metres / math.pi)
    future_radius_metres = initial_radius_metres + radial_spread_metres_per_hour * hours
    return math.pi * future_radius_metres**2 / 10_000