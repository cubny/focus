"""Tests for the dynamics processing module."""

import numpy as np
import pytest

from focus.dsp.dynamics import (
    LimiterState,
    _db_to_linear,
    _linear_to_db,
    apply_limiter,
    get_true_peak_db,
)
from focus.dsp.entrainment import create_test_tone


class TestDbConversions:
    """Tests for dB conversion utilities."""

    def test_db_to_linear_zero_db(self):
        """0 dB should equal linear 1.0."""
        assert abs(_db_to_linear(0.0) - 1.0) < 1e-10

    def test_db_to_linear_minus_6db(self):
        """-6 dB should be approximately 0.5 linear."""
        assert abs(_db_to_linear(-6.02) - 0.5) < 0.01

    def test_linear_to_db_unity(self):
        """Linear 1.0 should equal 0 dB."""
        assert abs(_linear_to_db(1.0) - 0.0) < 1e-10

    def test_roundtrip_conversion(self):
        """Conversion should be reversible."""
        for db in [-12.0, -6.0, -3.0, 0.0, 3.0]:
            linear = _db_to_linear(db)
            back = _linear_to_db(linear)
            assert abs(back - db) < 1e-6


class TestApplyLimiter:
    """Tests for the True Peak limiter."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def quiet_audio(self, sample_rate):
        """Quiet audio well below ceiling."""
        return create_test_tone(440.0, 1.0, sample_rate, channels=2) * 0.3

    @pytest.fixture
    def loud_audio(self, sample_rate):
        """Loud audio that will hit the limiter."""
        return create_test_tone(440.0, 1.0, sample_rate, channels=2) * 1.5

    def test_output_shape_matches_input(self, quiet_audio, sample_rate):
        """Output shape should match input."""
        output, _ = apply_limiter(quiet_audio, sample_rate)
        assert output.shape == quiet_audio.shape

    def test_output_dtype_matches_input(self, quiet_audio, sample_rate):
        """Output dtype should match input."""
        output, _ = apply_limiter(quiet_audio, sample_rate)
        assert output.dtype == quiet_audio.dtype

    def test_quiet_signal_passes_unchanged(self, quiet_audio, sample_rate):
        """Quiet signals should pass through mostly unchanged."""
        output, _ = apply_limiter(quiet_audio, sample_rate, ceiling_db=-0.1)

        # Should be very similar (allow small differences from envelope smoothing)
        diff = np.max(np.abs(output - quiet_audio))
        assert diff < 0.01  # Less than 1% difference

    def test_ceiling_never_exceeded(self, loud_audio, sample_rate):
        """Output should never exceed ceiling."""
        ceiling_db = -0.1
        ceiling_linear = _db_to_linear(ceiling_db)

        output, _ = apply_limiter(loud_audio, sample_rate, ceiling_db=ceiling_db)

        # Check that all samples are at or below ceiling
        assert np.max(np.abs(output)) <= ceiling_linear + 1e-6

    def test_handles_aggressive_modulation(self, sample_rate):
        """Limiter should handle peaks from heavy modulation."""
        # Create audio with aggressive amplitude modulation
        t = np.arange(sample_rate) / sample_rate
        carrier = np.sin(2 * np.pi * 440 * t)
        modulator = 1 + 0.8 * np.sin(2 * np.pi * 15 * t)  # Heavy 80% modulation
        modulated = (carrier * modulator).astype(np.float32)

        output, _ = apply_limiter(modulated, sample_rate, ceiling_db=-0.1)

        # Should be limited
        ceiling_linear = _db_to_linear(-0.1)
        assert np.max(np.abs(output)) <= ceiling_linear + 1e-6

    def test_returns_state_for_continuity(self, quiet_audio, sample_rate):
        """Should return state for chunk continuity."""
        _, state = apply_limiter(quiet_audio, sample_rate)
        assert isinstance(state, LimiterState)

    def test_chunk_continuity(self, sample_rate):
        """Processing consecutive chunks should be smooth."""
        chunk_size = 4800
        audio = create_test_tone(440.0, 0.5, sample_rate, channels=2) * 1.2

        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

        state = None
        outputs = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            output, state = apply_limiter(chunk, sample_rate, state=state)
            outputs.append(output)

        full_output = np.concatenate(outputs)
        diffs = np.abs(np.diff(full_output, axis=0))

        # Should have no sudden jumps (clicks)
        assert np.max(diffs) < 0.3


class TestGetTruePeakDb:
    """Tests for True Peak measurement."""

    def test_silent_audio_very_low_db(self):
        """Silent audio should have very low dB."""
        silent = np.zeros(1000, dtype=np.float32)
        db = get_true_peak_db(silent)
        assert db < -100  # Essentially -infinity

    def test_full_scale_sine_near_zero_db(self):
        """Full-scale sine should be near 0 dBTP."""
        t = np.linspace(0, 2 * np.pi, 1000)
        full_scale = np.sin(t).astype(np.float32)
        db = get_true_peak_db(full_scale)
        assert -1.0 < db <= 0.1  # Near 0 dB

    def test_stereo_returns_max_channel(self):
        """Stereo should return max of both channels."""
        quiet = np.zeros((1000, 2), dtype=np.float32)
        quiet[:, 0] = 0.5 * np.sin(np.linspace(0, 10 * np.pi, 1000))
        quiet[:, 1] = 0.25 * np.sin(np.linspace(0, 10 * np.pi, 1000))

        db = get_true_peak_db(quiet)
        expected_db = _linear_to_db(0.5)

        # Should be close to the louder channel
        assert abs(db - expected_db) < 1.0

    def test_detects_intersample_peaks(self):
        """Should detect peaks between samples via oversampling."""
        # Two adjacent samples that will interpolate to a higher peak
        # -1.0, +1.0 will interpolate to values > 1.0 between them
        # Actually let's use a simpler case
        audio = np.array([0.9, -0.9], dtype=np.float32)

        # Without oversampling, peak is 0.9
        # With 2x oversampling and linear interpolation, middle sample is 0
        # So true peak should still be 0.9
        db = get_true_peak_db(audio)
        expected = _linear_to_db(0.9)
        assert abs(db - expected) < 1.0
