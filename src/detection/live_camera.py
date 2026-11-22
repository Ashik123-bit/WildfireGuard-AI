from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from PIL import Image


def validate_live_source(source: int | str) -> int | str:
    if isinstance(source, int):
        if source < 0:
            raise ValueError("Camera index must be zero or greater.")
        return source

    try:
        parsed = urlsplit(source.strip())
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("Enter a valid RTSP or RTSPS stream URL.") from exc
    if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not hostname:
        raise ValueError("Enter a valid RTSP or RTSPS stream URL.")
    return source.strip()


class LiveCameraSource:
    def __init__(self, source: int | str) -> None:
        import cv2

        source = validate_live_source(source)
        try:
            if isinstance(source, str):
                timeout_parameters = [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    5000,
                ]
                self.capture = cv2.VideoCapture(
                    source, cv2.CAP_FFMPEG, timeout_parameters
                )
            else:
                self.capture = cv2.VideoCapture(source)
        except cv2.error as exc:
            raise RuntimeError("OpenCV could not initialize the camera stream.") from exc

        if not self.capture.isOpened():
            self.capture.release()
            raise ConnectionError(
                "Could not open the camera stream. Check the camera index, "
                "RTSP address, credentials, and network connection."
            )

    def read_frame(self) -> Image.Image | None:
        import cv2
        from PIL import Image

        try:
            ok, frame = self.capture.read()
        except cv2.error as exc:
            raise RuntimeError("OpenCV could not read from the camera stream.") from exc
        if not ok:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def close(self) -> None:
        self.capture.release()
