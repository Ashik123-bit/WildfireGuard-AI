from __future__ import annotations

import csv
import hashlib
import io
import math
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from src.detection.detector import Detection, VideoAnalysis, WildfireDetector
from src.detection.live_camera import LiveCameraSource, validate_live_source
from src.prediction.area_forecast import (
    is_in_training_region,
    predict_next_day_fire_area,
)
from src.prediction.spread_predictor import (
    bounding_box_coverage,
    estimate_visible_area_hectares,
)
from src.prediction.weather import fetch_weather_features
from src.response.risk_assessment import RiskAssessment, assess_detection_risk


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "wildfire_yolo_v2.pt"


@st.cache_resource
def load_detector(model_path: str) -> WildfireDetector:
    return WildfireDetector(model_path)


@st.cache_data(ttl=900)
def _cached_weather_features(
    latitude: float, longitude: float
) -> dict[str, float] | str:
    try:
        return fetch_weather_features(latitude, longitude)
    except (OSError, ValueError) as exc:
        return str(exc)


def _render_forecast_inputs() -> tuple[float | None, float | None, float | None]:
    st.sidebar.subheader("Automatic spread estimate")
    st.sidebar.caption(
        "Set the analysis location and ground footprint once. Only coordinates "
        "are sent to Open-Meteo; image and video data stay on this server."
    )

    def coordinate_input(
        label: str, key: str, minimum: float, maximum: float
    ) -> float | None:
        raw_value = st.sidebar.text_input(label, key=key, placeholder="Required")
        if not raw_value.strip():
            return None
        try:
            value = float(raw_value)
        except ValueError:
            st.sidebar.error(f"{label} must be a number.")
            return None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            st.sidebar.error(f"{label} must be between {minimum:g} and {maximum:g}.")
            return None
        return value

    latitude = coordinate_input("Latitude", "forecast_latitude", -90, 90)
    longitude = coordinate_input("Longitude", "forecast_longitude", -180, 180)
    footprint = st.sidebar.number_input(
        "Ground area visible in each scene (ha)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help=(
            "Required to convert visible fire boxes into a rough area proxy. "
            "Use 0 when unknown; smoke-box size is not burned area."
        ),
    )
    return latitude, longitude, footprint if footprint > 0 else None


def _render_auto_spread_forecast(
    current_area_hectares: float | None,
    latitude: float | None,
    longitude: float | None,
    source: str,
) -> None:
    st.subheader("Location-based next-day fire-growth estimate")
    if current_area_hectares is None:
        st.info(
            "Enter a calibrated ground footprint in the sidebar to convert "
            "fire-box pixels into an approximate area."
        )
        return
    if current_area_hectares <= 0:
        st.info(
            "No measurable fire-box area was detected; no area forecast was generated."
        )
        return
    if latitude is None or longitude is None:
        st.info(
            "Enter valid latitude and longitude in the sidebar for a weather-based estimate."
        )
        return
    if not is_in_training_region(latitude, longitude):
        st.warning(
            "No forecast was generated. The area model was trained on Canadian "
            "fire locations (approximately 43.6–69.3° N, 141.0–55.4° W)."
        )
        return

    weather = _cached_weather_features(latitude, longitude)
    if isinstance(weather, str):
        st.error(f"Could not retrieve weather for the forecast: {weather}")
        return

    try:
        forecast_area = predict_next_day_fire_area(
            current_area_hectares, weather, latitude, longitude
        )
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as exc:
        st.error(f"Spread estimate unavailable: {exc}")
        return

    st.markdown("#### Forecast inputs")
    input_columns = st.columns(3)
    input_columns[0].metric("Latitude", f"{latitude:.4f}°")
    input_columns[1].metric("Longitude", f"{longitude:.4f}°")
    input_columns[2].metric(
        "Current fire-box area (rough)", f"{current_area_hectares:.3f} ha"
    )

    st.markdown("#### Retrieved weather")
    weather_columns = st.columns(5)
    weather_columns[0].metric(
        "Daily max temperature", f"{weather['tmax']:.1f} °C"
    )
    weather_columns[1].metric("Precipitation", f"{weather['prec']:.1f} mm")
    weather_columns[2].metric("Wind speed", f"{weather['ws']:.1f} km/h")
    weather_columns[3].metric("Relative humidity", f"{weather['rh']:.0f}%")
    weather_columns[4].metric(
        "Vapor-pressure deficit", f"{weather['vpd']:.1f} hPa"
    )

    st.markdown("#### Next-day fire-growth estimate")
    forecast_columns = st.columns(2)
    forecast_columns[0].metric(
        "Current camera fire-box area (rough)", f"{current_area_hectares:.3f} ha"
    )
    forecast_columns[1].metric(
        "Predicted next-day fire growth", f"{forecast_area:.1f} ha"
    )
    st.bar_chart(
        [
            {
                "Measure": "Current camera fire-box area (rough)",
                "Area (ha)": current_area_hectares,
            },
            {
                "Measure": "Predicted next-day fire growth",
                "Area (ha)": forecast_area,
            },
        ],
        x="Measure",
        y="Area (ha)",
        color="#e8753d",
    )
    st.caption(
        f"Location-based regression using {source} detections, Open-Meteo weather, "
        "and historical Canadian wildfire records. The training target is the "
        "next recorded day's daily fire-area growth value; it is not a cumulative "
        "total burned area. The current value is a rough camera-derived fire-box "
        "visual estimate, not a verified real-world fire area, so the two "
        "chart values are not directly like-for-like. On a held-out fire-level "
        "test, the model's MAE was about 425 ha (R² 0.066). This experimental "
        "estimate is not validated outside its Canadian training region and is "
        "not for operational decisions."
    )


def _draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    colors = {"fire": "#ff4b4b", "smoke": "#f0a202"}
    for detection in detections:
        color = colors.get(detection.label, "#00c2a8")
        box = (detection.x1, detection.y1, detection.x2, detection.y2)
        draw.rectangle(box, outline=color, width=max(2, annotated.width // 300))
        draw.text(
            (detection.x1, max(0, detection.y1 - 20)),
            f"{detection.label} {detection.confidence:.0%}",
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
    return annotated


def _video_report_csv(analysis: VideoAnalysis) -> str:
    report = io.StringIO(newline="")
    writer = csv.writer(report)
    writer.writerow(
        ("timestamp_seconds", "detection", "confidence", "x1", "y1", "x2", "y2")
    )
    for frame in analysis.frames:
        if not frame.detections:
            writer.writerow(
                (f"{frame.timestamp_seconds:.3f}", "none", "", "", "", "", "")
            )
            continue
        for detection in frame.detections:
            writer.writerow(
                (
                    f"{frame.timestamp_seconds:.3f}",
                    detection.label,
                    f"{detection.confidence:.6f}",
                    f"{detection.x1:.2f}",
                    f"{detection.y1:.2f}",
                    f"{detection.x2:.2f}",
                    f"{detection.y2:.2f}",
                )
            )
    return report.getvalue()


def _render_video_analysis(
    analysis: VideoAnalysis,
    latitude: float | None,
    longitude: float | None,
    camera_footprint_hectares: float | None,
    confidence_threshold: float,
) -> None:
    detected_frames = analysis.detected_frames
    fire_frames = tuple(
        frame
        for frame in detected_frames
        if any(item.label == "fire" for item in frame.detections)
    )
    smoke_frames = tuple(
        frame
        for frame in detected_frames
        if any(item.label == "smoke" for item in frame.detections)
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric("Clip duration", f"{analysis.duration_seconds:.1f} s")
    summary_columns[1].metric("Frames analyzed", analysis.sampled_frame_count)
    summary_columns[2].metric("Frames with fire", len(fire_frames))
    summary_columns[3].metric("Frames with smoke", len(smoke_frames))
    st.download_button(
        "Download frame report (CSV)",
        data=_video_report_csv(analysis),
        file_name="wildfire_video_observations.csv",
        mime="text/csv",
        key="download_video_report",
    )
    if detected_frames:
        st.subheader("Observed activity (clip-wide)")
        for label, label_frames in (("Fire", fire_frames), ("Smoke", smoke_frames)):
            if label_frames:
                st.write(
                    f"{label}: first observed at "
                    f"{label_frames[0].timestamp_seconds:.1f} s; last observed at "
                    f"{label_frames[-1].timestamp_seconds:.1f} s."
                )
            else:
                st.write(f"{label}: not detected in sampled frames.")
        st.caption(
            "These are first/last sampled observations, not the actual ignition "
            "or extinguishment times."
        )
        timeline = [
            {
                "clip time (s)": round(frame.timestamp_seconds, 2),
                "fire boxes": sum(
                    item.label == "fire" for item in frame.detections
                ),
                "smoke boxes": sum(
                    item.label == "smoke" for item in frame.detections
                ),
            }
            for frame in analysis.frames
        ]
        st.line_chart(
            timeline,
            x="clip time (s)",
            y=["fire boxes", "smoke boxes"],
        )
        st.caption(
            "The smoke and fire observations above cover all sampled frames in "
            "the clip. Counts below apply only to the latest detected frame."
        )
        latest_detected_frame = detected_frames[-1]
        st.markdown(
            "#### Latest detected sampled frame · "
            f"{latest_detected_frame.timestamp_seconds:.1f} s"
        )
        latest_assessment = assess_detection_risk(
            latest_detected_frame.detections,
            analysis.image_width,
            analysis.image_height,
        )
        _render_detection_summary(
            list(latest_detected_frame.detections),
            latest_assessment,
            sum(
                item.label == "fire"
                for item in latest_detected_frame.detections
            ),
            sum(
                item.label == "smoke"
                for item in latest_detected_frame.detections
            ),
            confidence_threshold,
        )
    else:
        st.info(
            "No fire or smoke detections appeared in the sampled frames. "
            "This does not confirm the clip is clear: faint, distant, or "
            "unfamiliar smoke can be missed by this prototype."
        )
    latest_fire_frame = next(
        (
            frame
            for frame in reversed(analysis.frames)
            if any(item.label == "fire" for item in frame.detections)
        ),
        None,
    )
    if latest_fire_frame is None:
        st.info("No fire detections were available for a next-day fire-growth estimate.")
    elif camera_footprint_hectares is None:
        _render_auto_spread_forecast(None, latitude, longitude, "video")
    else:
        visible_area = estimate_visible_area_hectares(
            bounding_box_coverage(
                latest_fire_frame.detections,
                analysis.image_width,
                analysis.image_height,
            ),
            camera_footprint_hectares,
        )
        _render_auto_spread_forecast(visible_area, latitude, longitude, "video")


def _render_detections(
    image: Image.Image,
    detections: list[Detection],
    confidence_threshold: float,
    latitude: float | None,
    longitude: float | None,
    camera_footprint_hectares: float | None,
) -> None:
    assessment = assess_detection_risk(detections, image.width, image.height)
    fire_count = sum(item.label == "fire" for item in detections)
    smoke_count = sum(item.label == "smoke" for item in detections)

    st.image(_draw_detections(image, detections), use_container_width=True)
    _render_detection_summary(
        detections,
        assessment,
        fire_count,
        smoke_count,
        confidence_threshold,
    )

    st.subheader("Area estimate")
    st.caption(
        "Bounding boxes include background around flames. A physical area estimate "
        "requires the ground area visible in this camera frame."
    )
    visible_area = None
    if camera_footprint_hectares is not None:
        visible_area = estimate_visible_area_hectares(
            assessment.fire_coverage, camera_footprint_hectares
        )
        st.metric("Visible fire-box area (rough)", f"{visible_area:.3f} ha")
    else:
        st.info(
            "Set the ground area visible in the sidebar to convert image-space "
            "fire-box coverage into a rough area proxy."
        )

    _render_auto_spread_forecast(visible_area, latitude, longitude, "image")


def _render_detection_summary(
    detections: list[Detection],
    assessment: RiskAssessment,
    fire_count: int,
    smoke_count: int,
    confidence_threshold: float | None = None,
) -> None:
    fire_confidence = max(
        (item.confidence for item in detections if item.label == "fire"),
        default=0.0,
    )
    smoke_confidence = max(
        (item.confidence for item in detections if item.label == "smoke"),
        default=0.0,
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Risk assessment", assessment.level)
    metric_columns[1].metric(
        "Fire detections", fire_count, f"Highest confidence {fire_confidence:.0%}"
    )
    metric_columns[2].metric(
        "Smoke detections", smoke_count, f"Highest confidence {smoke_confidence:.0%}"
    )
    metric_columns[3].metric("Fire-box coverage", f"{assessment.fire_coverage:.1%}")

    status_renderers = {
        "No detection": st.success,
        "Detected": st.info,
        "Elevated": st.warning,
        "High": st.error,
        "Uncertain": st.warning,
    }
    render_status = status_renderers.get(assessment.level, st.info)
    recommendation = assessment.recommendation
    status_message = (
        f"**Why this level:** {assessment.note}  \n"
        f"**Advisory:** {recommendation}"
    )
    if confidence_threshold is not None:
        status_message += f"  \n**Detection threshold:** {confidence_threshold:.0%}"
    render_status(status_message)

    if smoke_count == 0:
        st.info(
            "No smoke boxes were detected. This model can miss smoke; verify the "
            "scene visually."
        )

    if detections:
        st.markdown("#### Detected objects")
        st.table(
            [
                {
                    "Class": item.label.title(),
                    "Confidence": f"{item.confidence:.1%}",
                }
                for item in detections
            ]
        )
    else:
        st.caption("No fire or smoke objects were returned by the detector.")


def _stop_live_monitor() -> None:
    live_source = st.session_state.pop("live_source", None)
    if live_source is not None:
        live_source.close()
    st.session_state.pop("live_observations", None)


def _render_page_chrome() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1380px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .wg-hero {
            background: linear-gradient(120deg, #10271e 0%, #24553b 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 22px;
            color: #f7fbf7;
            margin-bottom: 1.4rem;
            padding: 2rem 2.2rem;
        }
        .wg-hero-label {
            color: #c3e3cb;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            margin: 0 0 0.5rem;
        }
        .wg-hero h1 {
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3rem);
            letter-spacing: -0.04em;
            margin: 0;
        }
        .wg-hero p {
            color: #e0eee4;
            font-size: 1rem;
            margin: 0.6rem 0 0;
            max-width: 52rem;
        }
        div[data-testid="stAlert"] {
            border-radius: 12px;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(35, 75, 52, 0.12);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            min-height: 7rem;
        }
        [data-testid="stTabs"] [role="tab"] {
            border-radius: 10px 10px 0 0;
            font-weight: 650;
        }
        [data-testid="stSubheader"] {
            margin-top: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="wg-hero">
            <p class="wg-hero-label">VISUAL MONITORING · DECISION SUPPORT</p>
            <h1>WildfireGuard AI</h1>
            <p>Review possible fire and smoke signals from photos, video clips,
            and live camera feeds.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=1)
def _render_live_monitor(
    detector: WildfireDetector,
    confidence_threshold: float,
    latitude: float | None,
    longitude: float | None,
    camera_footprint_hectares: float | None,
) -> None:
    live_source = st.session_state.get("live_source")
    if live_source is None:
        st.info("Start a camera stream above to begin monitoring.")
        return

    try:
        image = live_source.read_frame()
        if image is None:
            live_source.close()
            st.session_state.pop("live_source", None)
            st.error(
                "The stream stopped returning frames. Check the camera and "
                "restart monitoring."
            )
            return
        detections = detector.predict(image, confidence_threshold)
    except (OSError, RuntimeError, ValueError) as exc:
        live_source.close()
        st.session_state.pop("live_source", None)
        st.error(f"Live analysis stopped: {exc}")
        return

    now = datetime.now().astimezone()
    observations = st.session_state.setdefault("live_observations", {})
    detected_labels = {item.label for item in detections}
    for label in ("fire", "smoke"):
        if label in detected_labels:
            observations.setdefault(f"{label}_first", now)
            observations[f"{label}_last"] = now

    assessment = assess_detection_risk(detections, image.width, image.height)
    st.image(_draw_detections(image, detections), use_container_width=True)
    _render_detection_summary(
        detections,
        assessment,
        sum(item.label == "fire" for item in detections),
        sum(item.label == "smoke" for item in detections),
        confidence_threshold,
    )

    for label in ("fire", "smoke"):
        first_seen = observations.get(f"{label}_first")
        last_seen = observations.get(f"{label}_last")
        if first_seen is None:
            st.write(f"{label.title()}: not observed during this monitoring session")
        else:
            st.write(
                f"{label.title()} first observed: {first_seen.strftime('%H:%M:%S')} "
                f"| last observed: {last_seen.strftime('%H:%M:%S')}"
            )
    st.caption(
        "First/last observed times are not confirmed ignition or extinguishment times."
    )

    if camera_footprint_hectares is not None:
        visible_area = estimate_visible_area_hectares(
            assessment.fire_coverage, camera_footprint_hectares
        )
        st.metric("Visible fire-box area (rough)", f"{visible_area:.3f} ha")
        _render_auto_spread_forecast(
            visible_area, latitude, longitude, "live camera"
        )
    else:
        _render_auto_spread_forecast(
            None, latitude, longitude, "live camera"
        )


def main() -> None:
    st.set_page_config(
        page_title="WildfireGuard AI",
        page_icon="🔥",
        layout="wide",
    )
    _render_page_chrome()
    st.warning(
        "Decision-support prototype only. It can miss fires or raise false alarms. "
        "Area and spread outputs are rough scenarios, not evacuation guidance. "
        "Recommendations are advisory; for an emergency, follow local guidance "
        "and contact relevant local emergency authorities."
    )

    if not MODEL_PATH.is_file():
        st.info(
            "The detector has not been trained yet. Install dependencies, then run "
            "`python train.py` from the project folder. Training uses the bundled "
            "annotations and saves `models/wildfire_yolo_v2.pt`."
        )
        with st.expander("Training options"):
            st.code(
                "pip install -r requirements.txt\n"
                "python train.py --epochs 10 --image-size 416 --batch 4 --device cpu\n"
                "streamlit run app.py",
                language="powershell",
            )
        return

    try:
        detector = load_detector(str(MODEL_PATH))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not load the trained detector: {exc}")
        return

    confidence_threshold = st.sidebar.slider(
        "Detection confidence",
        min_value=0.01,
        max_value=0.90,
        value=0.10,
        step=0.01,
        help=(
            "Lower values may surface weaker fire or smoke candidates, but false "
            "alarms increase. This prototype can still miss smoke."
        ),
    )
    latitude, longitude, camera_footprint_hectares = _render_forecast_inputs()
    image_tab, video_tab, live_tab = st.tabs(
        ["Image analysis", "Video analysis", "Live camera/CCTV"]
    )

    with image_tab:
        st.subheader("Analyze an image")
        st.caption(
            "Upload a scene to review candidate fire and smoke regions. "
            "A missing box is not proof that the scene is clear."
        )
        uploaded_image = st.file_uploader(
            "Choose a wildfire scene image",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
        )
        if uploaded_image is not None:
            try:
                image = Image.open(uploaded_image).convert("RGB")
                detections = detector.predict(image, confidence_threshold)
                _render_detections(
                    image,
                    detections,
                    confidence_threshold,
                    latitude,
                    longitude,
                    camera_footprint_hectares,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Image analysis failed: {exc}")

    with video_tab:
        st.subheader("Review a video clip")
        st.caption(
            "Sampled frames are processed in small batches to reduce inference "
            "overhead. Shorter intervals inspect more frames and take longer."
        )
        uploaded_video = st.file_uploader(
            "Choose a video clip",
            type=["mp4", "mov", "avi", "mkv"],
            key="video_upload",
        )
        sample_interval = st.slider(
            "Analyze one frame every (seconds)",
            min_value=0.25,
            max_value=5.0,
            value=1.0,
            step=0.25,
        )
        if uploaded_video is None:
            st.session_state.pop("video_analysis_result", None)
        else:
            video_bytes = uploaded_video.getvalue()
            video_key = (
                hashlib.sha256(video_bytes).hexdigest(),
                sample_interval,
                confidence_threshold,
            )
            if st.button("Analyze video"):
                st.session_state.pop("video_analysis_result", None)
                temporary_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=Path(uploaded_video.name).suffix, delete=False
                    ) as temporary_file:
                        temporary_file.write(video_bytes)
                        temporary_path = Path(temporary_file.name)
                    analysis = detector.analyze_video(
                        temporary_path,
                        sample_interval_seconds=sample_interval,
                        confidence_threshold=confidence_threshold,
                    )
                    st.session_state["video_analysis_result"] = (video_key, analysis)
                except (ImportError, OSError, ValueError, RuntimeError) as exc:
                    st.error(f"Video analysis failed: {exc}")
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)

            stored_result = st.session_state.get("video_analysis_result")
            if stored_result is not None and stored_result[0] == video_key:
                _render_video_analysis(
                    stored_result[1],
                    latitude,
                    longitude,
                    camera_footprint_hectares,
                    confidence_threshold,
                )
            elif stored_result is not None:
                st.info("The clip or settings changed. Analyze again to refresh results.")

    with live_tab:
        st.subheader("Live fire and smoke monitoring")
        st.caption(
            "The stream is opened by the computer running this WildfireGuard "
            "server. Frames are analyzed live and are not recorded or saved."
        )
        if st.session_state.get("live_source") is None:
            source_type = st.radio(
                "Camera source",
                ["Webcam on this computer", "CCTV RTSP/RTSPS stream"],
                horizontal=True,
                key="live_source_type",
            )
            if source_type == "Webcam on this computer":
                camera_index = st.number_input(
                    "Camera index",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    help="Try 0 for the default camera, then 1 or 2 for another.",
                )
                source_input: int | str = int(camera_index)
            else:
                source_input = st.text_input(
                    "CCTV stream URL",
                    type="password",
                    placeholder="rtsp://camera-address/stream",
                    help="Credentials are kept in this session and are not shown in the UI.",
                    key="live_rtsp_url",
                )

            if st.button("Start live monitoring", type="primary"):
                try:
                    source = validate_live_source(source_input)
                    st.session_state["live_source"] = LiveCameraSource(source)
                    st.session_state["live_observations"] = {}
                    st.rerun()
                except (ConnectionError, ImportError, OSError, ValueError) as exc:
                    st.error(f"Could not start live monitoring: {exc}")
        else:
            if st.button("Stop live monitoring"):
                _stop_live_monitor()
                st.rerun()
            _render_live_monitor(
                detector,
                confidence_threshold,
                latitude,
                longitude,
                camera_footprint_hectares,
            )

if __name__ == "__main__":
    main()