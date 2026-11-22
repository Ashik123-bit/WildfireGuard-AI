from __future__ import annotations

import csv
import io
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app import _render_detection_summary, _video_report_csv
from src.data.prepare_dataset import (
    _load_voc_labels,
    convert_yolo_labels,
    prepare_dataset,
)
from src.detection.detector import (
    INFERENCE_BATCH_SIZE,
    Detection,
    VideoAnalysis,
    VideoFrame,
    WildfireDetector,
)
from src.detection.live_camera import validate_live_source
from src.prediction.area_forecast import (
    is_in_training_region,
    predict_next_day_fire_area,
)
from src.prediction.spread_predictor import (
    bounding_box_coverage,
    estimate_visible_area_hectares,
    project_spread_hectares,
)
from src.prediction.weather import (
    _vapor_pressure_deficit_hpa,
    fetch_weather_features,
)
from src.response.risk_assessment import assess_detection_risk


class SpreadPredictionTests(unittest.TestCase):
    def test_zero_smoke_detections_show_missed_smoke_caution(self) -> None:
        with (
            patch("app.st.columns", return_value=[Mock() for _ in range(4)]),
            patch("app.st.success"),
            patch("app.st.info") as show_info,
            patch("app.st.caption"),
        ):
            _render_detection_summary(
                [], assess_detection_risk([], 100, 100), 0, 0
            )

        self.assertIn("No smoke boxes were detected", show_info.call_args.args[0])

    def test_coverage_unions_overlapping_fire_boxes(self) -> None:
        boxes = [
            Detection("fire", 0.9, 0, 0, 6, 10),
            Detection("fire", 0.8, 4, 0, 10, 10),
            Detection("smoke", 0.7, 0, 0, 10, 10),
        ]

        self.assertAlmostEqual(bounding_box_coverage(boxes, 10, 10), 1.0)
        self.assertAlmostEqual(
            bounding_box_coverage(boxes, 10, 10, label="smoke"), 1.0
        )
        self.assertAlmostEqual(
            bounding_box_coverage(boxes[:2], 10, 10, label="smoke"), 0.0
        )

    def test_area_estimate_and_spread_scenario(self) -> None:
        visible_area = estimate_visible_area_hectares(0.25, 4.0)
        self.assertEqual(visible_area, 1.0)
        expected_area = 49 * math.pi / 10_000
        self.assertAlmostEqual(
            project_spread_hectares(math.pi / 10_000, 2.0, 3.0),
            expected_area,
        )
        self.assertAlmostEqual(project_spread_hectares(1.0, 0.0, 6.0), 1.0)
        self.assertEqual(project_spread_hectares(0.0, 0.0, 4.0), 0.0)

    def test_area_inputs_reject_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            estimate_visible_area_hectares(1.2, 1.0)
        with self.assertRaises(ValueError):
            project_spread_hectares(1.0, -1.0, 2.0)

    def test_area_forecast_converts_log_prediction_and_rejects_out_of_region(self) -> None:
        class FakeModel:
            def predict(self, _: object) -> list[float]:
                return [math.log1p(25.0)]

        with patch(
            "src.prediction.area_forecast.load_area_forecast_model",
            return_value=FakeModel(),
        ):
            prediction = predict_next_day_fire_area(
                2.0,
                {"tmax": 25.0, "vpd": 10.0, "ws": 8.0, "rh": 40.0, "prec": 0.0},
                56.0,
                -112.0,
            )

        self.assertAlmostEqual(prediction, 25.0)
        self.assertTrue(is_in_training_region(56.0, -112.0))
        self.assertFalse(is_in_training_region(40.0, -112.0))
        with self.assertRaises(ValueError):
            predict_next_day_fire_area(
                2.0,
                {"tmax": 25.0, "vpd": 10.0, "ws": 8.0, "rh": 40.0, "prec": 0.0},
                40.0,
                -112.0,
            )

    def test_weather_vpd_uses_humidity_and_temperature(self) -> None:
        self.assertAlmostEqual(
            _vapor_pressure_deficit_hpa(25.0, 50.0),
            15.84,
            delta=0.1,
        )

    def test_weather_features_are_parsed_from_api_response(self) -> None:
        response = io.BytesIO(
            json.dumps(
                {
                    "current": {
                        "temperature_2m": 20.0,
                        "relative_humidity_2m": 50,
                        "wind_speed_10m": 12.0,
                    },
                    "daily": {
                        "temperature_2m_max": [24.0],
                        "precipitation_sum": [1.5],
                    },
                }
            ).encode("utf-8")
        )
        with patch("src.prediction.weather.urlopen", return_value=response):
            weather = fetch_weather_features(56.0, -112.0)

        self.assertEqual(weather["tmax"], 24.0)
        self.assertEqual(weather["prec"], 1.5)
        self.assertEqual(weather["ws"], 12.0)
        self.assertEqual(weather["rh"], 50.0)
        self.assertGreater(weather["vpd"], 0)

    def test_empty_frame_is_not_mislabeled_as_uncertain_detection(self) -> None:
        assessment = assess_detection_risk([], 100, 100)
        self.assertEqual(assessment.level, "No detection")
        self.assertEqual(assessment.recommendation, "Continue routine monitoring.")

    def test_smoke_only_detection_is_flagged(self) -> None:
        smoke = Detection("smoke", 0.8, 1, 1, 10, 10)
        assessment = assess_detection_risk([smoke], 100, 100)
        self.assertEqual(assessment.level, "Elevated")
        self.assertEqual(
            assessment.recommendation,
            "Increase monitoring and seek human verification.",
        )

    def test_risk_recommendations_match_assessment_levels(self) -> None:
        cases = (
            (
                [Detection("fire", 0.8, 0, 0, 10, 10)],
                "Detected",
                "Verify the detection.",
            ),
            (
                [Detection("fire", 0.8, 0, 0, 50, 50)],
                "High",
                "Promptly notify relevant local emergency authorities.",
            ),
            (
                [Detection("fire", 0.2, 0, 0, 10, 10)],
                "Uncertain",
                "Require human verification.",
            ),
        )
        for detections, expected_level, expected_recommendation in cases:
            with self.subTest(level=expected_level):
                assessment = assess_detection_risk(detections, 100, 100)
                self.assertEqual(assessment.level, expected_level)
                self.assertEqual(
                    assessment.recommendation, expected_recommendation
                )


class DatasetPreparationTests(unittest.TestCase):
    def test_yolo_labels_preserve_fire_and_smoke_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "labels.txt"
            label_path.write_text(
                "0 0.5 0.5 0.2 0.4\n1 0.2 0.3 0.1 0.1\n",
                encoding="utf-8",
            )
            labels = convert_yolo_labels(label_path)
        self.assertEqual(len(labels), 2)
        self.assertTrue(labels[0].startswith("0 "))
        self.assertTrue(labels[1].startswith("1 "))

    def test_yolo_labels_reject_non_finite_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "labels.txt"
            label_path.write_text("0 nan 0.5 0.2 0.4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                convert_yolo_labels(label_path)

    def test_voc_boxes_convert_to_normalized_yolo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_path = Path(temporary_directory) / "fire.xml"
            root = ET.Element("annotation")
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = "100"
            ET.SubElement(size, "height").text = "200"
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = "fire"
            box = ET.SubElement(obj, "bndbox")
            for name, value in (
                ("xmin", "10"),
                ("ymin", "20"),
                ("xmax", "50"),
                ("ymax", "100"),
            ):
                ET.SubElement(box, name).text = value
            ET.ElementTree(root).write(annotation_path)

            labels = _load_voc_labels(annotation_path)

        self.assertEqual(labels, ("0 0.300000 0.300000 0.400000 0.400000",))


class VideoAnalysisTests(unittest.TestCase):
    def test_video_frames_are_inferred_in_bounded_batches(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.frames_remaining = INFERENCE_BATCH_SIZE + 2

            def isOpened(self) -> bool:
                return True

            def get(self, _: int) -> float:
                return 4.0

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.frames_remaining == 0:
                    return False, None
                self.frames_remaining -= 1
                return True, np.zeros((16, 16, 3), dtype=np.uint8)

            def release(self) -> None:
                pass

        class FakeResult:
            boxes = None

        class RecordingModel:
            names = {0: "fire", 1: "smoke"}

            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def predict(self, source: list[object], **_: object) -> list[FakeResult]:
                self.batch_sizes.append(len(source))
                return [FakeResult() for _ in source]

        detector = object.__new__(WildfireDetector)
        model = RecordingModel()
        detector.model = model

        with patch("cv2.VideoCapture", return_value=FakeCapture()):
            analysis = detector.analyze_video(
                "unused.mp4", sample_interval_seconds=0.25
            )

        self.assertEqual(analysis.sampled_frame_count, INFERENCE_BATCH_SIZE + 2)
        self.assertEqual(model.batch_sizes, [INFERENCE_BATCH_SIZE, 2])
        self.assertEqual((analysis.image_width, analysis.image_height), (16, 16))

    def test_video_csv_report_includes_detections_and_empty_frames(self) -> None:
        analysis = VideoAnalysis(
            frames=(
                VideoFrame(
                    0.25,
                    (Detection("smoke", 0.8, 1, 2, 3, 4),),
                ),
                VideoFrame(0.5, ()),
            ),
            duration_seconds=0.5,
            sampled_frame_count=2,
        )

        rows = list(csv.DictReader(io.StringIO(_video_report_csv(analysis))))

        self.assertEqual(rows[0]["detection"], "smoke")
        self.assertEqual(rows[0]["timestamp_seconds"], "0.250")
        self.assertEqual(rows[1]["detection"], "none")
        self.assertEqual(rows[1]["timestamp_seconds"], "0.500")

    def test_live_source_accepts_camera_index_and_rtsp_but_not_other_schemes(self) -> None:
        self.assertEqual(validate_live_source(0), 0)
        self.assertEqual(
            validate_live_source("rtsp://camera.local/live"),
            "rtsp://camera.local/live",
        )
        with self.assertRaises(ValueError):
            validate_live_source("http://camera.local/live")
        with self.assertRaises(ValueError):
            validate_live_source(-1)

    def test_prepare_dataset_creates_consistent_train_and_validation_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory) / "project"
            images_root = (
                project_root / "CCTV_Fire_Smoke_Emergency_Detection_Dataset" / "images"
            )
            labels_root = (
                project_root / "CCTV_Fire_Smoke_Emergency_Detection_Dataset" / "labels"
            )
            images_root.mkdir(parents=True, exist_ok=True)
            labels_root.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                (images_root / f"scene_{index}.png").write_bytes(b"test image")
                (labels_root / f"scene_{index}.txt").write_text(
                    f"{index % 2} 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )

            output_root = Path(temporary_directory) / "processed"
            stale_image = output_root / "images" / "train" / "stale.jpg"
            stale_label = output_root / "labels" / "train" / "stale.txt"
            stale_cache = output_root / "labels" / "train.cache"
            stale_image.parent.mkdir(parents=True)
            stale_label.parent.mkdir(parents=True)
            stale_cache.parent.mkdir(parents=True, exist_ok=True)
            stale_image.write_bytes(b"stale")
            stale_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            stale_cache.write_text("stale cache", encoding="utf-8")

            yaml_path, train_count, val_count = prepare_dataset(
                project_root, output_root, validation_fraction=0.25
            )

            self.assertEqual((train_count, val_count), (3, 1))
            self.assertFalse(stale_image.exists())
            self.assertFalse(stale_label.exists())
            self.assertFalse(stale_cache.exists())
            self.assertTrue(yaml_path.is_file())
            for split, expected_count in (("train", 3), ("val", 1)):
                images = list((output_root / "images" / split).iterdir())
                labels = list((output_root / "labels" / split).iterdir())
                self.assertEqual(len(images), expected_count)
                self.assertEqual(len(labels), expected_count)
                self.assertEqual(
                    {image.stem for image in images},
                    {label.stem for label in labels},
                )

    def test_prepare_dataset_includes_nested_external_yolo_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory) / "project"
            dataset_root = (
                project_root
                / "data"
                / "external"
                / "fire-and-smoke-synthetic-10k"
            )
            for split, class_id in (("train", 0), ("train", 1), ("val", 1)):
                images_root = dataset_root / "images" / split
                labels_root = dataset_root / "labels" / split
                images_root.mkdir(parents=True, exist_ok=True)
                labels_root.mkdir(parents=True, exist_ok=True)
                image_name = f"scene_{split}_{class_id}.png"
                (images_root / image_name).write_bytes(b"test image")
                (labels_root / Path(image_name).with_suffix(".txt")).write_text(
                    f"{class_id} 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )

            output_root = Path(temporary_directory) / "processed"
            _, train_count, val_count = prepare_dataset(
                project_root, output_root, validation_fraction=0.5
            )

            self.assertEqual((train_count, val_count), (1, 1))
            output_labels = [
                label.read_text(encoding="utf-8").split()[0]
                for split in ("train", "val")
                for label in (output_root / "labels" / split).glob("*.txt")
            ]
            self.assertCountEqual(output_labels, ["0", "1"])


if __name__ == "__main__":
    unittest.main()
