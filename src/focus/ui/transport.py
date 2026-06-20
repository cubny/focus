"""Single-key transport controls and a live status line.

These sit on top of one shared :class:`PlaybackState`. The audio loop reads the
flags each iteration; the keyboard reader mutates them. Keeping the input source
(raw stdin here) decoupled from the state means other frontends (e.g. OS media
keys) could drive the same object later without touching the audio loop.

Unix only (macOS/Linux/WSL): uses ``termios``/``tty`` raw input and the asyncio
reader on stdin. Callers must gate construction on ``sys.stdin.isatty()``.
"""

import asyncio
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass

VOLUME_STEP = 0.1
MAX_KEY_READ_BYTES = 3  # Enough for arrow-key escape sequences handled by _iter_key_events.


def _iter_key_events(data: bytes):
    while data:
        if data.startswith((b"\x1b[A", b"\x1b[B")):
            yield data[:3]
            data = data[3:]
        else:
            yield data[:1]
            data = data[1:]


@dataclass
class PlaybackState:
    """Shared, mutable state between the keyboard reader and the audio loop."""

    # Control flags (set by the keyboard reader, consumed by the audio loop)
    paused: bool = False
    skip_requested: bool = False
    quit_requested: bool = False
    show_help: bool = False
    volume: float = 1.0

    # Display fields (set by the audio loop, read by the status line)
    profile_name: str = ""
    modulation_freq: float = 0.0
    modulation_depth: float = 0.0
    elapsed_seconds: float = 0.0
    buffer_seconds: float = 0.0
    status: str = "connecting"  # connecting | playing | paused | reconnecting

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def volume_up(self) -> None:
        self.volume = min(1.0, round(self.volume + VOLUME_STEP, 2))

    def volume_down(self) -> None:
        self.volume = max(0.0, round(self.volume - VOLUME_STEP, 2))

    def handle_key(self, data: bytes) -> None:
        """Apply a key (or escape sequence) read from the terminal."""
        if data in (b" ", b"p", b"P"):
            self.toggle_pause()
        elif data in (b"n", b"N"):
            self.skip_requested = True
        elif data in (b"q", b"Q"):
            self.quit_requested = True
        elif data == b"?":
            self.show_help = not self.show_help
        elif data in (b"+", b"=", b"\x1b[A"):  # '=' is unshifted '+'; \x1b[A is up arrow
            self.volume_up()
        elif data in (b"-", b"_", b"\x1b[B"):  # \x1b[B is down arrow
            self.volume_down()


class KeyboardController:
    """Reads single keys from stdin (cbreak mode) and mutates a PlaybackState.

    Uses ``tty.setcbreak`` rather than raw mode so ``Ctrl+C`` still raises
    ``KeyboardInterrupt`` and acts as a hard stop. The terminal is always
    restored in :meth:`stop`, including on error.
    """

    def __init__(
        self,
        state: PlaybackState,
        loop: asyncio.AbstractEventLoop | None = None,
        fd: int | None = None,
        on_change: Callable[[], None] | None = None,
    ):
        self.state = state
        self._loop = loop
        self._fd = sys.stdin.fileno() if fd is None else fd
        self._old_settings = None
        self._active = False
        # Called after each keypress mutates the state, so the UI updates
        # immediately rather than waiting for the next audio chunk.
        self._on_change = on_change

    def start(self) -> None:
        import termios
        import tty

        self._loop = self._loop or asyncio.get_running_loop()
        self._old_settings = termios.tcgetattr(self._fd)
        # Mark active and save settings *before* the risky add_reader so that
        # stop() always restores the terminal even if registration fails.
        self._active = True
        tty.setcbreak(self._fd)
        try:
            self._loop.add_reader(self._fd, self._on_readable)
        except Exception:
            self.stop()
            raise

    def _on_readable(self) -> None:
        try:
            data = os.read(self._fd, MAX_KEY_READ_BYTES)
        except (OSError, BlockingIOError):
            return
        for key in _iter_key_events(data):
            self.state.handle_key(key)
            if self._on_change is not None:
                self._on_change()

    def stop(self) -> None:
        import termios

        if not self._active:
            return
        self._active = False
        if self._loop is not None:
            try:
                self._loop.remove_reader(self._fd)
            except Exception:
                pass
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None


def _format_time(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


_STATUS_TOKENS = {
    "connecting": "… connecting",
    "playing": "▸ playing",
    "paused": "⏸ paused",
    "reconnecting": "… reconnecting",
}


class StatusLine:
    """A single-line, in-place status display with inline key hints."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self._active = False

    def start(self) -> None:
        self._active = True

    def render(self, state: PlaybackState) -> None:
        if not self._active:
            return
        line = self._format(state)
        # Truncate to one less than the terminal width (cosmetic right-edge
        # cleanup). This counts code points, not display columns, so glyphs that
        # render 2 wide (♫, ⏸, ↑↓ on some terminals) can still overflow — hence
        # the autowrap guard below is what actually keeps us on one row.
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        if len(line) > width - 1:
            line = line[: width - 1]
        # Disable the terminal's auto-wrap (DECAWM) for the repaint: an over-long
        # line then clamps at the last column instead of wrapping onto a second
        # physical row. Without this, the cursor lands on the wrapped row and the
        # next "\r\x1b[2K" clears only that row, leaving a trail of stale lines.
        # Re-enable autowrap immediately after.
        self.stream.write("\x1b[?7l\r\x1b[2K" + line + "\x1b[?7h")
        self.stream.flush()

    def finish(self) -> None:
        """Move off the status line so following output starts cleanly."""
        if self._active:
            # Restore autowrap (in case the last render left it disabled) and
            # clear the status row so following output starts cleanly.
            self.stream.write("\r\x1b[2K\x1b[?7h")
            self.stream.flush()
            self._active = False

    @staticmethod
    def _format(s: PlaybackState) -> str:
        token = _STATUS_TOKENS.get(s.status, s.status)
        vol = f"{int(round(s.volume * 100))}%"
        info = (
            f"♫ {s.profile_name} · {s.modulation_freq:.0f}Hz @ {s.modulation_depth:.0%}"
            f"   {token}   {_format_time(s.elapsed_seconds)}"
            f"   vol {vol}   buf {s.buffer_seconds:.1f}s"
        )
        if s.show_help:
            hints = (
                "[space/p] pause  [n] next take  [↑↓ or +/-] volume  "
                "[?] hide help  [q] quit  [Ctrl+C] stop"
            )
        else:
            hints = "[space] pause  [n] next  [↑↓] volume  [?] help  [q] quit"
        return f"{info}   ·   {hints}"
