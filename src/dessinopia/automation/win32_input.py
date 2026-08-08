from __future__ import annotations

import ctypes
import math
import sys
import threading
import time
from collections.abc import Callable
from ctypes import wintypes

from dessinopia.core.models import Point


class InputInjectionError(RuntimeError):
    pass


if sys.platform == "win32":
    ULONG_PTR = wintypes.WPARAM

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        )

    class _INPUTUNION(ctypes.Union):
        _fields_ = (("mi", MOUSEINPUT),)

    class INPUT(ctypes.Structure):
        _anonymous_ = ("union",)
        _fields_ = (("type", wintypes.DWORD), ("union", _INPUTUNION))


MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
INPUT_MOUSE = 0
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
VK_F12 = 0x7B


class _EmergencyKeyWatcher:
    """Poll and latch even a short emergency-key press.

    A vision check can keep the automation thread busy for close to a second.
    Looking only at the current high bit of ``GetAsyncKeyState`` afterwards can
    miss a key that was pressed and released during that interval. The watcher
    polls independently and keeps the event sticky until input cleanup.
    """

    def __init__(self, poll_key_state: Callable[[], int], *, interval: float = 0.008) -> None:
        self._poll_key_state = poll_key_state
        self._interval = interval
        self._requested = threading.Event()
        self._closed = threading.Event()
        # Consume an old low-order transition bit without hiding a key that is
        # already physically down when the runner starts.
        if int(self._poll_key_state()) & 0x8000:
            self._requested.set()
        self._thread = threading.Thread(
            target=self._watch,
            name="Dessinopia-F12",
            daemon=True,
        )
        self._thread.start()

    def requested(self) -> bool:
        self._sample()
        return self._requested.is_set()

    def close(self) -> None:
        self._closed.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=0.1)

    def _watch(self) -> None:
        while not self._closed.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        # High bit: currently down. Low bit: pressed since the preceding poll.
        if int(self._poll_key_state()) & 0x8001:
            self._requested.set()


def _normalize_absolute_coordinate(value: float, origin: int, extent: int) -> int:
    """Convertit un pixel du bureau virtuel en coordonnée SendInput.

    Une calibration invalide ne doit jamais être rabattue silencieusement sur un
    bord de l'écran : ce bord peut contenir la barre des tâches ou une autre
    application. Refuser l'injection est le comportement sûr.
    """
    if extent < 2:
        raise InputInjectionError("Les dimensions du bureau virtuel sont invalides")
    if not math.isfinite(value):
        raise InputInjectionError("La calibration contient une coordonnée non finie")
    maximum = origin + extent - 1
    if value < origin or value > maximum:
        raise InputInjectionError("La calibration cible une position située hors du bureau virtuel")
    return round((value - origin) * 65535 / (extent - 1))


class Win32Mouse:
    def __init__(self, checkpoint: Callable[[], None] | None = None) -> None:
        if sys.platform != "win32":
            raise InputInjectionError(
                "Le pilotage de la souris est disponible uniquement sous Windows"
            )
        self.user32 = ctypes.windll.user32
        self.user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self.user32.SendInput.restype = wintypes.UINT
        self.user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
        self.user32.GetAsyncKeyState.restype = wintypes.SHORT
        self.checkpoint = checkpoint or (lambda: None)
        self._left_is_down = False
        self._emergency = _EmergencyKeyWatcher(
            lambda: int(self.user32.GetAsyncKeyState(VK_F12))
        )

    def emergency_stop_requested(self) -> bool:
        return self._emergency.requested()

    @property
    def left_is_down(self) -> bool:
        return self._left_is_down

    def _send(self, flags: int, x: float = 0.0, y: float = 0.0) -> None:
        virtual_left = self.user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        virtual_top = self.user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        virtual_width = self.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        virtual_height = self.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        normalized_x = _normalize_absolute_coordinate(x, virtual_left, virtual_width)
        normalized_y = _normalize_absolute_coordinate(y, virtual_top, virtual_height)
        mouse = MOUSEINPUT(
            dx=normalized_x,
            dy=normalized_y,
            mouseData=0,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        item = INPUT(type=INPUT_MOUSE, union=_INPUTUNION(mi=mouse))
        if self.user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(INPUT)) != 1:
            raise InputInjectionError(
                "Windows a refusé l'injection. Heartopia et Dessinopia doivent "
                "avoir le même niveau de privilèges."
            )

    def move_to(self, point: Point, duration: float = 0.0) -> None:
        self.checkpoint()
        start_x, start_y = self.position()
        distance = math.hypot(point.x - start_x, point.y - start_y)
        if distance < 0.5:
            return
        # Even callers requesting an immediate move get a short visible glide.
        # This prevents cursor teleportation while keeping adjacent-cell drags
        # precise and responsive.
        minimum_duration = min(0.12, max(0.012, distance / 6000))
        duration = max(float(duration), minimum_duration)
        steps = max(2, min(120, int(max(duration * 100, distance / 10))))
        started = time.monotonic()
        for step in range(1, steps + 1):
            self.checkpoint()
            ratio = step / steps
            eased = ratio * ratio * (3 - 2 * ratio)
            x = start_x + (point.x - start_x) * eased
            y = start_y + (point.y - start_y) * eased
            self._send(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, x, y)
            target_time = started + duration * ratio
            remaining = target_time - time.monotonic()
            if remaining > 0:
                time.sleep(min(remaining, 0.02))

    def position(self) -> tuple[int, int]:
        point = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(point))
        return int(point.x), int(point.y)

    def left_down(self) -> None:
        x, y = self.position()
        self._send(MOUSEEVENTF_LEFTDOWN, x, y)
        self._left_is_down = True

    def left_up(self) -> None:
        if not self._left_is_down:
            return
        x, y = self.position()
        self._send(MOUSEEVENTF_LEFTUP, x, y)
        self._left_is_down = False

    def click(self) -> None:
        self.left_down()
        time.sleep(0.018)
        self.left_up()

    def release_all(self) -> None:
        try:
            self.left_up()
        except InputInjectionError:
            self._left_is_down = False
        finally:
            self._emergency.close()
