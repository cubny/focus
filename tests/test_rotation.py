"""Tests for the seamless session-rotation helpers in the Lyria client."""

import numpy as np

from focus.generation.lyria_client import (
    CROSSFADE_OUT_END_FRAC,
    _ChunkBuffer,
    _crossfade_gains,
)


class TestCrossfadeGains:
    """Crossfade gain curves."""

    def test_endpoints(self):
        # Start of the crossfade: full old, no new.
        fade_out, fade_in = _crossfade_gains(0, 1, 1000)
        assert fade_out[0] == 1.0
        assert abs(fade_in[0]) < 1e-9

        # End of the crossfade: no old, full new.
        fade_out, fade_in = _crossfade_gains(1000, 1, 1000)
        assert abs(fade_out[0]) < 1e-9
        assert abs(fade_in[0] - 1.0) < 1e-9

    def test_outgoing_fades_out_before_the_end(self):
        """Default curve silences the outgoing track by out_end_frac (no ghost)."""
        total = 1000
        fade_out, _ = _crossfade_gains(0, total, total)
        end_idx = int(CROSSFADE_OUT_END_FRAC * total)
        # Outgoing has reached (near) silence by the cutover point...
        assert fade_out[end_idx] < 1e-6
        # ...and stays there for the rest of the crossfade.
        assert np.all(fade_out[end_idx:] < 1e-6)
        # ...while it was still audible earlier.
        assert fade_out[end_idx // 2] > 0.3

    def test_symmetric_is_equal_power(self):
        """With out_end_frac=1.0 the curve is a symmetric equal-power crossfade."""
        total = 4096
        fade_out, fade_in = _crossfade_gains(0, total, total, out_end_frac=1.0)
        power = fade_out**2 + fade_in**2
        np.testing.assert_allclose(power, 1.0, atol=1e-9)

    def test_monotonic_and_contiguous(self):
        # Two consecutive segments should tile the curve monotonically.
        fo_a, fi_a = _crossfade_gains(0, 100, 300)
        fo_b, fi_b = _crossfade_gains(100, 100, 300)
        assert np.all(np.diff(fo_a) <= 1e-12)  # fade-out non-increasing
        assert np.all(np.diff(fi_a) >= -1e-12)  # fade-in non-decreasing
        assert fo_a[-1] >= fo_b[0]
        assert fi_a[-1] <= fi_b[0]


class TestChunkBuffer:
    """Sample-accurate FIFO used to align two live streams."""

    def _stereo(self, start, n):
        col = np.arange(start, start + n, dtype=np.float32)
        return np.column_stack([col, col])

    def test_take_across_chunk_boundaries(self):
        buf = _ChunkBuffer()
        buf.add(self._stereo(0, 3))
        buf.add(self._stereo(3, 4))
        assert buf.total == 7

        first = buf.take(5)
        assert first.shape == (5, 2)
        np.testing.assert_array_equal(first[:, 0], np.arange(0, 5))
        assert buf.total == 2

        rest = buf.take(10)  # asking for more than available
        assert rest.shape == (2, 2)
        np.testing.assert_array_equal(rest[:, 0], np.arange(5, 7))
        assert buf.total == 0

    def test_take_empty_returns_zero_length(self):
        buf = _ChunkBuffer()
        out = buf.take(4)
        assert out.shape == (0, 2)

    def test_add_ignores_empty(self):
        buf = _ChunkBuffer()
        buf.add(None)
        buf.add(np.zeros((0, 2), dtype=np.float32))
        assert buf.total == 0
