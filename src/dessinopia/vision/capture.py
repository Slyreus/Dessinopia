from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterable
from ctypes import wintypes
from dataclasses import dataclass, replace

import mss
import numpy as np

from dessinopia.core.models import CalibrationProfile, Point, ScreenRect


@dataclass(frozen=True, slots=True)
class WindowInfo:
    handle: int
    title: str
    rect: ScreenRect


def _client_bounds(hwnd: int) -> ScreenRect | None:
    rect = wintypes.RECT()
    origin = wintypes.POINT()
    user32 = ctypes.windll.user32
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)) or not user32.ClientToScreen(
        hwnd, ctypes.byref(origin)
    ):
        return None
    return ScreenRect(origin.x, origin.y, rect.right, rect.bottom)


@dataclass(frozen=True, slots=True)
class NativeMonitor:
    """Physical desktop coordinates reported by Windows."""

    name: str
    rect: ScreenRect
    primary: bool = False


def monitor_containing_point(monitors: Iterable[ScreenRect], point: Point) -> ScreenRect | None:
    """Return the physical monitor containing *point*, including negative origins."""

    for monitor in monitors:
        if monitor.left <= point.x < monitor.right and monitor.top <= point.y < monitor.bottom:
            return monitor
    return None


def localize_profile(
    profile: CalibrationProfile,
    capture_bounds: ScreenRect,
    capture_width: int,
    capture_height: int,
) -> CalibrationProfile:
    """Express a desktop calibration in coordinates local to one capture.

    A translated profile keeps the dimensions of the image on which it was
    calibrated. Resetting them to the actual capture prevents downstream vision
    code from scaling already-physical desktop coordinates a second time.
    """

    if capture_width <= 0 or capture_height <= 0:
        raise ValueError("La capture doit avoir des dimensions positives")
    localized = profile.translated(-capture_bounds.left, -capture_bounds.top)
    return replace(
        localized,
        reference_width=int(capture_width),
        reference_height=int(capture_height),
    )


def native_monitors() -> tuple[NativeMonitor, ...]:
    """Enumerate Win32 monitors with stable display names and physical bounds.

    Qt exposes screen geometry in device-independent pixels, while mouse injection and
    ``mss`` use physical virtual-desktop pixels.  Keeping the native monitor origin is
    essential for mixed-DPI layouts and for monitors placed left/above the primary one.
    """

    if sys.platform != "win32":
        return ()

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = (
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        )

    result: list[NativeMonitor] = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HANDLE,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    @callback_type
    def callback(
        monitor_handle: int,
        _device_context: int,
        _monitor_rect: ctypes.POINTER(wintypes.RECT),
        _data: int,
    ) -> bool:
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        if ctypes.windll.user32.GetMonitorInfoW(monitor_handle, ctypes.byref(info)):
            rect = info.rcMonitor
            result.append(
                NativeMonitor(
                    name=str(info.szDevice),
                    rect=ScreenRect(
                        float(rect.left),
                        float(rect.top),
                        float(rect.right - rect.left),
                        float(rect.bottom - rect.top),
                    ),
                    primary=bool(info.dwFlags & 1),
                )
            )
        return True

    ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0)
    return tuple(result)


class HeartopiaWindowLocator:
    TITLES = ("heartopia", "heartbeat town", "artopia", "心动小镇")

    @classmethod
    def is_candidate(cls, title: str, process_id: int) -> bool:
        """Match only a known game title owned by a different process."""

        return process_id != os.getpid() and title.strip().casefold() in cls.TITLES

    def find(self) -> WindowInfo | None:
        if sys.platform != "win32":
            return None
        user32 = ctypes.windll.user32
        matches: list[WindowInfo] = []

        enum_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @enum_callback
        def callback(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.IsIconic(hwnd):
                return True
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            # Une correspondance partielle attraperait facilement un navigateur ouvert
            # sur un guide « Heartopia ». Les titres connus sont courts et le repli par
            # détection d'écran reste disponible si une future version les modifie.
            if not self.is_candidate(title, int(process_id.value)):
                return True
            bounds = _client_bounds(hwnd)
            if bounds and bounds.width > 0 and bounds.height > 0:
                matches.append(WindowInfo(int(hwnd), title, bounds))
            return True

        user32.EnumWindows(callback, 0)
        if not matches:
            return None
        return max(matches, key=lambda item: item.rect.width * item.rect.height)


class ScreenCapture:
    @staticmethod
    def grab_window(window: WindowInfo) -> np.ndarray | None:
        """Read one frame from the game itself, excluding overlapping windows."""
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtGui import QImage
        from PySide6.QtMultimedia import QMediaCaptureSession, QVideoFrame, QVideoSink, QWindowCapture

        capture = QWindowCapture()
        matches = [
            item for item in capture.capturableWindows() if item.description() == window.title
        ]
        if len(matches) != 1:
            return None
        session = QMediaCaptureSession()
        sink = QVideoSink()
        session.setWindowCapture(capture)
        session.setVideoSink(sink)
        capture.setWindow(matches[0])
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        capture.errorOccurred.connect(loop.quit)

        def frame_ready(frame: QVideoFrame) -> None:
            if frame.isValid():
                loop.quit()

        sink.videoFrameChanged.connect(frame_ready)
        try:
            timer.start(1800)
            capture.start()
            if not sink.videoFrame().isValid() and capture.error() == QWindowCapture.Error.NoError:
                loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            image = sink.videoFrame().toImage()
            if image.isNull() or _client_bounds(window.handle) != window.rect:
                return None
            width, height = int(window.rect.width), int(window.rect.height)
            if (image.width(), image.height()) != (width, height):
                # WGC may include the title bar and visible frame in windowed mode.
                frame = wintypes.RECT()
                hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                    window.handle, 9, ctypes.byref(frame), ctypes.sizeof(frame)
                )
                x, y = int(window.rect.left - frame.left), int(window.rect.top - frame.top)
                if (
                    hr
                    or (frame.right - frame.left, frame.bottom - frame.top)
                    != (image.width(), image.height())
                    or min(x, y) < 0
                    or x + width > image.width()
                    or y + height > image.height()
                ):
                    return None
                image = image.copy(x, y, width, height)
            image = image.convertToFormat(QImage.Format.Format_RGB888)
            rows = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine())
            return rows[:, :width * 3].reshape(height, width, 3).copy()
        finally:
            timer.stop()
            capture.stop()

    @staticmethod
    def _rect_from_monitor(monitor: dict[str, int]) -> ScreenRect:
        return ScreenRect(
            float(monitor["left"]),
            float(monitor["top"]),
            float(monitor["width"]),
            float(monitor["height"]),
        )

    def virtual_bounds(self) -> ScreenRect:
        """Bounds of the complete physical virtual desktop, origin included."""

        with mss.MSS() as session:
            return self._rect_from_monitor(session.monitors[0])

    def monitor_bounds(self) -> tuple[ScreenRect, ...]:
        """Physical bounds for individual monitors (the primary is normally first)."""

        with mss.MSS() as session:
            return tuple(self._rect_from_monitor(item) for item in session.monitors[1:])

    def grab(self, rect: ScreenRect | None = None) -> np.ndarray:
        with mss.MSS() as session:
            if rect is None:
                monitor = session.monitors[0]
            else:
                monitor = {
                    "left": int(round(rect.left)),
                    "top": int(round(rect.top)),
                    "width": max(1, int(round(rect.width))),
                    "height": max(1, int(round(rect.height))),
                }
            bgra = np.asarray(session.grab(monitor), dtype=np.uint8)
        return bgra[..., [2, 1, 0]].copy()
