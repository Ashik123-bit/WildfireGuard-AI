from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


INFERENCE_BATCH_SIZE = 16


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area_pixels(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class VideoFrame:
    timestamp_seconds: float
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class VideoAnalysis:
    frames: tuple[VideoFrame, ...]
    duration_seconds: float
    sampled_frame_count: int
    image_width: int = 0
    image_height: int = 0

    @property
    def detected_frames(self) -> tuple[VideoFrame, ...]:
        return tuple(frame for frame in self.frames if frame.detections)


class WildfireDetector:
    def __init__(self, model_path: str | Path) -> None:
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Trained model not found at {model_file}. Run train.py first."
            )

        from ultralytics import YOLO

        self.model: Any = YOLO(str(model_file))

    def _detections_from_result(self, result: Any) -> list[Detection]:
        if result.boxes is None:
            return []

        names = self.model.names
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = str(names[class_id]).strip().lower()
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf.item()),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def predict(
        self, image: Image.Image, confidence_threshold: float = 0.10
    ) -> list[Detection]:
        result = self.model.predict(
            source=image.convert("RGB"),
            conf=confidence_threshold,
            verbose=False,
        )[0]
        return self._detections_from_result(result)

    def _predict_batch(
        self, images: list[Image.Image], confidence_threshold: float
    ) -> list[list[Detection]]:
        results = self.model.predict(
            source=images,
            conf=confidence_threshold,
            verbose=False,
        )
        if len(results) != len(images):
            raise RuntimeError(
                "The detector returned a different number of results than input frames."
            )
        return [self._detections_from_result(result) for result in results]

    def analyze_video(
        self,
        video_path: str | Path,
        sample_interval_seconds: float = 1.0,
        confidence_threshold: float = 0.10,
    ) -> VideoAnalysis:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be greater than zero.")

        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise ValueError("Video does not report a valid frame rate.")

        frame_step = max(1, round(fps * sample_interval_seconds))
        frames: list[VideoFrame] = []
        pending_images: list[Image.Image] = []
        pending_timestamps: list[float] = []
        frame_index = 0
        last_timestamp = 0.0
        image_width = 0
        image_height = 0

        def flush_batch() -> None:
            if not pending_images:
                return
            batch_detections = self._predict_batch(
                pending_images, confidence_threshold
            )
            frames.extend(
                VideoFrame(timestamp, tuple(detections))
                for timestamp, detections in zip(
                    pending_timestamps, batch_detections, strict=True
                )
            )
            pending_images.clear()
            pending_timestamps.clear()

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if image_width == 0 or image_height == 0:
                    image_height, image_width = frame.shape[:2]
                if frame_index % frame_step == 0:
                    timestamp = frame_index / fps
                    last_timestamp = timestamp
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pending_images.append(Image.fromarray(rgb_frame))
                    pending_timestamps.append(timestamp)
                    if len(pending_images) >= INFERENCE_BATCH_SIZE:
                        flush_batch()
                frame_index += 1
            flush_batch()
        finally:
            capture.release()

        if frame_index == 0:
            raise ValueError("Video contains no readable frames.")

        duration = max(last_timestamp, (frame_index - 1) / fps)
        return VideoAnalysis(
            frames=tuple(frames),
            duration_seconds=duration,
            sampled_frame_count=len(frames),
            image_width=image_width,
            image_height=image_height,
        )