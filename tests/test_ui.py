"""Tests for the interactive transport controls and launcher gating."""

import asyncio
import io
import os
import time

try:
    import pty
except ImportError:  # pragma: no cover - Windows
    pty = None

import pytest
from click.testing import CliRunner

from focus.audio.output import SOUNDDEVICE_AVAILABLE, AudioOutput, MockAudioOutput
from focus.cli import main
from focus.ui import launcher
from focus.ui.transport import KeyboardController, PlaybackState, StatusLine, _format_time

# The pty-backed tests exercise the raw terminal readers; pty is Unix-only.
requires_pty = pytest.mark.skipif(
    not hasattr(os, "openpty"), reason="pty is unavailable on this platform"
)


class TestPlaybackState:
    def test_space_and_p_toggle_pause(self):
        s = PlaybackState()
        s.handle_key(b" ")
        assert s.paused is True
        s.handle_key(b"p")
        assert s.paused is False

    def test_n_requests_skip(self):
        s = PlaybackState()
        s.handle_key(b"n")
        assert s.skip_requested is True

    def test_q_requests_quit(self):
        s = PlaybackState()
        s.handle_key(b"q")
        assert s.quit_requested is True

    def test_question_toggles_help(self):
        s = PlaybackState()
        s.handle_key(b"?")
        assert s.show_help is True
        s.handle_key(b"?")
        assert s.show_help is False

    def test_volume_keys_and_arrows(self):
        s = PlaybackState(volume=0.5)
        s.handle_key(b"+")
        assert s.volume == 0.6
        s.handle_key(b"-")
        assert s.volume == 0.5
        s.handle_key(b"\x1b[A")  # up arrow
        assert s.volume == 0.6
        s.handle_key(b"\x1b[B")  # down arrow
        assert s.volume == 0.5

    def test_volume_clamped_to_unit_range(self):
        s = PlaybackState(volume=1.0)
        s.handle_key(b"+")
        assert s.volume == 1.0
        s = PlaybackState(volume=0.0)
        s.handle_key(b"-")
        assert s.volume == 0.0

    def test_unknown_key_is_ignored(self):
        s = PlaybackState()
        s.handle_key(b"z")
        assert s == PlaybackState()


class TestStatusLine:
    def test_format_time(self):
        assert _format_time(0) == "00:00"
        assert _format_time(74) == "01:14"

    def test_format_includes_profile_and_hints(self):
        s = PlaybackState(profile_name="deep-work", modulation_freq=18.0, status="playing")
        line = StatusLine._format(s)
        assert "deep-work" in line
        assert "18Hz" in line
        assert "[q] quit" in line

    def test_help_toggle_changes_hints(self):
        s = PlaybackState(show_help=True)
        assert "next take" in StatusLine._format(s)

    def test_render_truncates_to_terminal_width(self, monkeypatch):
        # The rendered payload must stay under the terminal width so writing it
        # never triggers auto-wrap (which would spawn a new line per update).
        monkeypatch.setenv("COLUMNS", "40")
        s = PlaybackState(profile_name="adhd-support", modulation_freq=15.0, status="playing")
        buf = io.StringIO()
        line = StatusLine(stream=buf)
        line.start()
        line.render(s)
        payload = (
            buf.getvalue()
            .replace("\x1b[?7l", "")
            .replace("\x1b[?7h", "")
            .replace("\r", "")
            .replace("\x1b[2K", "")
        )
        assert len(payload) <= 39

    def test_render_brackets_repaint_with_autowrap_toggle(self):
        # The repaint must disable autowrap (DECAWM) and re-enable it, so an
        # over-long line clamps at the last column instead of wrapping onto a
        # second physical row (which would leave a trail of stale status lines).
        s = PlaybackState(profile_name="deep-work", status="playing")
        buf = io.StringIO()
        line = StatusLine(stream=buf)
        line.start()
        line.render(s)
        out = buf.getvalue()
        assert out.startswith("\x1b[?7l")  # autowrap off before the repaint
        assert out.endswith("\x1b[?7h")  # autowrap restored after

    def test_finish_restores_autowrap(self):
        buf = io.StringIO()
        line = StatusLine(stream=buf)
        line.start()
        line.finish()
        assert "\x1b[?7h" in buf.getvalue()


@requires_pty
class TestKeyboardOnChange:
    def test_on_change_fires_after_keypress(self):
        master, slave = pty.openpty()
        state = PlaybackState(volume=0.5)
        calls = []

        async def run():
            kc = KeyboardController(state, fd=slave, on_change=lambda: calls.append(state.volume))
            kc.start()
            try:
                os.write(master, b"+")  # volume up
                await asyncio.sleep(0.1)
            finally:
                kc.stop()

        try:
            asyncio.run(run())
        finally:
            os.close(master)
            os.close(slave)

        assert calls == [0.6]  # 0.5 + one step, reported instantly via on_change


class TestMockAudioOutputControls:
    def test_set_volume_clamps(self):
        out = MockAudioOutput()
        out.set_volume(2.0)
        assert out.volume == 1.0
        out.set_volume(-1.0)
        assert out.volume == 0.0

    def test_pause_resume(self):
        out = MockAudioOutput()
        out.pause()
        assert out._paused is True
        out.resume()
        assert out._paused is False


class TestAudioOutputControls:
    @pytest.mark.skipif(
        not SOUNDDEVICE_AVAILABLE, reason="sounddevice/PortAudio is unavailable in this environment"
    )
    def test_set_volume_clamps(self):
        out = AudioOutput()
        out.set_volume(2.0)
        assert out.volume == 1.0
        out.set_volume(-1.0)
        assert out.volume == 0.0

    @pytest.mark.skipif(
        not SOUNDDEVICE_AVAILABLE, reason="sounddevice/PortAudio is unavailable in this environment"
    )
    def test_pause_blocks_stream_start(self):
        # While paused, the stream must not (re)start even if buffer fills.
        out = AudioOutput()
        out.start()
        out.pause()
        out._maybe_start_stream()
        assert out._stream_active is False


@requires_pty
class TestTerminalReaders:
    """Exercise the real os.read paths over a pty (otherwise never run headless).

    ``_getch`` flushes pending input when it enters cbreak mode (so stray
    keystrokes typed before a prompt are dropped), so the test must write the
    key only *after* the reader is blocked in ``read`` — hence the helper thread.
    """

    def _getch_with_input(self, data: bytes) -> bytes:
        import threading

        master, slave = pty.openpty()
        result = {}

        def reader():
            result["key"] = launcher._getch(fd=slave)

        t = threading.Thread(target=reader)
        t.start()
        try:
            time.sleep(0.1)  # let _getch flush and block in read()
            os.write(master, data)
            t.join(timeout=2.0)
        finally:
            os.close(master)
            os.close(slave)
        assert not t.is_alive(), "_getch did not return"
        return result["key"]

    def test_getch_reads_full_escape_sequence(self):
        assert self._getch_with_input(b"\x1b[A") == b"\x1b[A"  # up arrow

    def test_getch_reads_single_key(self):
        assert self._getch_with_input(b"q") == b"q"

    def test_keyboard_controller_dispatches_keys(self):
        master, slave = pty.openpty()
        state = PlaybackState()

        async def run():
            kc = KeyboardController(state, fd=slave)
            kc.start()  # flushes pending input here
            try:
                os.write(master, b" ")  # pause — written after the flush
                await asyncio.sleep(0.1)
            finally:
                kc.stop()

        try:
            asyncio.run(run())
        finally:
            os.close(master)
            os.close(slave)

        assert state.paused is True


class TestLauncherGating:
    def test_bare_invocation_without_tty_shows_help(self):
        # CliRunner provides a non-TTY stdin/stdout, so the bare invocation must
        # fall back to help text rather than entering the interactive launcher.
        result = CliRunner().invoke(main, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_profiles_subcommand_still_works(self):
        result = CliRunner().invoke(main, ["profiles"])
        assert result.exit_code == 0
        assert "deep-work" in result.output
