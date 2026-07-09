"""Tests for the real-time spectrum analyzer and terminal visualizer."""

import io

import numpy as np
import pytest
from click.testing import CliRunner

from focus.analysis.realtime import SpectrumAnalyzer
from focus.audio.output import SOUNDDEVICE_AVAILABLE, AudioOutput
from focus.cli import main
from focus.ui.spectrum import SpectrumDisplay, SpectrumRenderer
from focus.ui.transport import PlaybackState

requires_sounddevice = pytest.mark.skipif(
    not SOUNDDEVICE_AVAILABLE, reason="sounddevice/PortAudio unavailable"
)


def _sine(freq: float, sr: int = 48000, n: int = 24000, amp: float = 0.8) -> np.ndarray:
    t = np.arange(n) / sr
    mono = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.column_stack([mono, mono])


class TestSpectrumAnalyzer:
    def test_output_shape_and_range(self):
        a = SpectrumAnalyzer()
        a.push(_sine(1000))
        v = a.compute(60)
        assert len(v) == 60
        assert float(v.min()) >= 0.0
        assert float(v.max()) <= 1.0

    def test_tone_peaks_in_matching_band(self):
        a = SpectrumAnalyzer()
        a.push(_sine(1000))
        for _ in range(6):  # let the fast attack settle
            v = a.compute(60)
        peak = int(v.argmax())
        assert v[peak] > 0.7
        # The peak band's FFT-bin range must actually cover 1 kHz.
        lo, hi = a._band_bins[peak]
        band_freqs = a._freqs[lo:hi]
        assert band_freqs.min() <= 1000.0 <= band_freqs.max() + a._freqs[1]
        # A far-away low band should be well below the peak.
        assert v[0] < v[peak] - 0.4

    def test_push_shifts_window_in_place_for_small_chunks(self):
        # Chunks smaller than fft_size shift the rolling window left in place
        # (an overlapping slice copy); the newest samples must land at the end.
        a = SpectrumAnalyzer(fft_size=8)
        a.push(np.arange(1, 6, dtype=np.float32))  # 5 samples into an 8-window
        assert np.array_equal(a._buf, [0, 0, 0, 1, 2, 3, 4, 5])
        a.push(np.arange(6, 9, dtype=np.float32))  # 3 more, oldest fall off
        assert np.array_equal(a._buf, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_push_keeps_tail_of_oversized_chunk(self):
        a = SpectrumAnalyzer(fft_size=4)
        a.push(np.arange(10, dtype=np.float32))
        assert np.array_equal(a._buf, [6, 7, 8, 9])

    def test_silence_is_near_zero(self):
        a = SpectrumAnalyzer()
        a.push_silence()
        for _ in range(20):
            v = a.compute(40)
        assert float(v.max()) < 1e-3

    def test_slow_decay_after_drop(self):
        a = SpectrumAnalyzer(decay=0.1)
        a.push(_sine(1000))
        for _ in range(6):
            high = a.compute(60).copy()
        peak = int(high.argmax())
        a.push_silence()
        after_one = a.compute(60)
        # One frame of silence must not collapse the bar; it decays gradually.
        assert after_one[peak] >= high[peak] - 0.1 - 1e-6
        assert after_one[peak] > 0.0

    def test_rise_is_fast(self):
        a = SpectrumAnalyzer(attack=0.6)
        a.push_silence()
        a.compute(60)
        a.push(_sine(1000))
        one = a.compute(60)
        peak = int(one.argmax())
        # A single frame of attack should already be a large fraction of target.
        assert one[peak] > 0.4


class TestSpectrumRenderer:
    def test_full_column_is_all_blocks(self):
        rows = SpectrumRenderer.render_rows(np.array([1.0]), 5)
        assert rows == ["█"] * 5

    def test_empty_column_is_all_spaces(self):
        rows = SpectrumRenderer.render_rows(np.array([0.0]), 5)
        assert rows == [" "] * 5

    def test_bars_rise_from_the_bottom(self):
        rows = SpectrumRenderer.render_rows(np.array([0.5]), 4)
        # Top rows empty, bottom rows full (rises from the bottom).
        assert rows[0] == " "
        assert rows[-1] == "█"

    def test_row_width_matches_band_count(self):
        rows = SpectrumRenderer.render_rows(np.array([0.2, 0.5, 0.9]), 3)
        assert all(len(r) == 3 for r in rows)


class TestSpectrumDisplay:
    def test_render_brackets_and_status_and_cursor_up(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        monkeypatch.setenv("LINES", "24")
        state = PlaybackState(profile_name="deep-work", status="playing")
        analyzer = SpectrumAnalyzer()
        analyzer.push(_sine(1000))
        buf = io.StringIO()
        display = SpectrumDisplay(state, analyzer, stream=buf, height=8)
        display.start()
        display.render()
        out = buf.getvalue()
        assert "\x1b[?25l" in out  # cursor hidden on start
        assert "\x1b[?7l" in out and "\x1b[?7h" in out  # autowrap toggled
        assert "deep-work" in out  # status row reuses format_status_line
        # Second render moves the cursor up over the previously drawn block.
        buf.truncate(0)
        buf.seek(0)
        display.render()
        # 8 spectrum rows joined to the status row => 8 newlines => move up 8.
        assert "\x1b[8A" in buf.getvalue()

    def test_finish_restores_cursor_and_autowrap(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        monkeypatch.setenv("LINES", "24")
        state = PlaybackState(profile_name="deep-work")
        display = SpectrumDisplay(state, SpectrumAnalyzer(), stream=io.StringIO())
        display.start()
        display.render()
        display.stream.truncate(0)
        display.stream.seek(0)
        display.finish()
        out = display.stream.getvalue()
        assert "\x1b[?25h" in out  # cursor restored
        assert "\x1b[?7h" in out  # autowrap restored

    def test_short_terminal_falls_back_to_status_only(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        monkeypatch.setenv("LINES", "3")
        state = PlaybackState(profile_name="deep-work", status="playing")
        buf = io.StringIO()
        display = SpectrumDisplay(state, SpectrumAnalyzer(), stream=buf)
        display.start()
        display.render()
        # Only the status row is drawn (no preceding spectrum rows).
        assert display._prev_lines == 0
        assert "deep-work" in buf.getvalue()


class TestSourceDrivenRender:
    def test_render_pulls_a_fresh_window_from_the_source_each_frame(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        monkeypatch.setenv("LINES", "24")
        calls = {"n": 0}

        def source(n):
            calls["n"] += 1
            return _sine(1000)[:, 0]  # mono window

        display = SpectrumDisplay(
            PlaybackState(profile_name="x"),
            SpectrumAnalyzer(),
            stream=io.StringIO(),
            source=source,
        )
        display.start()
        display.render()
        display.render()
        assert calls["n"] == 2  # one pull per frame, so bars track live audio


@requires_sounddevice
class TestAudioOutputVisualizerTap:
    def test_latest_samples_returns_recent_played_audio(self):
        o = AudioOutput()
        for i in range(6):
            o._capture_visualizer(np.full((2048, 2), float(i), dtype=np.float32))
        tail = o.latest_samples(2048)
        assert len(tail) == 2048
        assert np.all(tail == 5.0)  # most recent block

    def test_ring_wraps_around(self):
        o = AudioOutput()
        ramp = np.arange(3000, dtype=np.float32)
        for _ in range(3):  # 9000 samples into an 8192 ring forces a wrap
            o._capture_visualizer(np.column_stack([ramp, ramp]))
        last = o.latest_samples(500)
        assert np.all(last == np.arange(2500, 3000))

    def test_pause_clears_the_tap(self):
        o = AudioOutput()
        o._capture_visualizer(np.ones((2048, 2), dtype=np.float32))
        o.pause()
        assert np.all(o.latest_samples(2048) == 0.0)  # bars fall while paused


class TestCliGate:
    @staticmethod
    def _invoke(monkeypatch, args):
        """Run the CLI with the session stubbed out (no audio device, no stream)."""
        captured = {}

        async def fake_run_session(profile, use_mock, duration, output_path=None, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("focus.cli._run_session", fake_run_session)
        result = CliRunner().invoke(main, args)
        return result, captured

    def test_spectrum_flag_threads_through(self, monkeypatch):
        result, captured = self._invoke(monkeypatch, ["start", "--mock", "--spectrum"])
        assert result.exit_code == 0
        assert captured["spectrum"] is True

    def test_no_spectrum_flag_threads_through(self, monkeypatch):
        result, captured = self._invoke(monkeypatch, ["start", "--mock", "--no-spectrum"])
        assert result.exit_code == 0
        assert captured["spectrum"] is False

    def test_spectrum_defaults_on(self, monkeypatch):
        _, captured = self._invoke(monkeypatch, ["start", "--mock"])
        assert captured["spectrum"] is True

    def test_no_ansi_block_in_non_tty(self, monkeypatch):
        # CliRunner is not a tty, so the visualizer must never hide the cursor
        # or emit its in-place block into piped output.
        result, _ = self._invoke(monkeypatch, ["start", "--mock", "--spectrum"])
        assert "\x1b[?25l" not in result.output
        assert "\x1b[J" not in result.output

    def test_duration_below_minimum_fast_fails(self):
        # Guards the real fast-fail path; no session (and no audio) is started.
        result = CliRunner().invoke(main, ["start", "--mock", "--duration", "59"])
        assert result.exit_code != 0
        assert "at least 60 seconds" in result.output
