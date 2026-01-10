"""Tests for the spatial audio processing module."""

import numpy as np
import pytest

from focus.dsp.entrainment import create_test_tone
from focus.dsp.spatial import (
    ReverbState,
    apply_reverb,
    apply_stereo_widening,
)


class TestApplyReverb:
    """Tests for the Schroeder reverb implementation."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def mono_audio(self, sample_rate):
        """1 second of mono sine wave at 440 Hz."""
        return create_test_tone(440.0, 1.0, sample_rate, channels=1)

    @pytest.fixture
    def stereo_audio(self, sample_rate):
        """1 second of stereo sine wave at 440 Hz."""
        return create_test_tone(440.0, 1.0, sample_rate, channels=2)

    def test_output_shape_matches_input_mono(self, mono_audio, sample_rate):
        """Mono output shape should match input."""
        output, _ = apply_reverb(mono_audio, sample_rate)
        assert output.shape == mono_audio.shape

    def test_output_shape_matches_input_stereo(self, stereo_audio, sample_rate):
        """Stereo output shape should match input."""
        output, _ = apply_reverb(stereo_audio, sample_rate)
        assert output.shape == stereo_audio.shape

    def test_output_dtype_matches_input(self, mono_audio, sample_rate):
        """Output dtype should match input."""
        output, _ = apply_reverb(mono_audio, sample_rate)
        assert output.dtype == mono_audio.dtype

    def test_zero_mix_returns_dry_signal(self, mono_audio, sample_rate):
        """With wet_dry_mix=0, output should equal input."""
        output, _ = apply_reverb(mono_audio, sample_rate, wet_dry_mix=0.0)
        np.testing.assert_array_almost_equal(output, mono_audio, decimal=5)

    def test_returns_state_for_continuity(self, mono_audio, sample_rate):
        """Should return state object for chunk continuity."""
        _, state = apply_reverb(mono_audio, sample_rate)
        assert isinstance(state, ReverbState)

    def test_chunk_continuity_no_clicks(self, sample_rate):
        """Processing consecutive chunks should not introduce clicks."""
        chunks = [create_test_tone(440.0, 0.1, sample_rate, channels=1) for _ in range(5)]

        state = None
        outputs = []
        for chunk in chunks:
            output, state = apply_reverb(chunk, sample_rate, state=state)
            outputs.append(output)

        # Concatenate and check for discontinuities
        full_output = np.concatenate(outputs)
        diffs = np.abs(np.diff(full_output))

        # Maximum diff should be reasonable (no clicks)
        assert np.max(diffs) < 0.2  # Threshold for detecting clicks

    def test_reverb_adds_energy_over_time(self, sample_rate):
        """Reverb should add decaying energy after impulse."""
        # Create impulse
        impulse = np.zeros(sample_rate, dtype=np.float32)
        impulse[0] = 1.0

        # Use higher wet mix and larger room for measurable tail
        output, _ = apply_reverb(impulse, sample_rate, wet_dry_mix=1.0, room_size=0.8)

        # Output should have energy beyond the initial impulse
        # Check that there's some energy in the early reflections region
        early_energy = np.mean(np.abs(output[1000:5000]))  # ~20-100ms
        assert early_energy > 0.0001  # Some reverb present


class TestApplyStereoWidening:
    """Tests for the Mid-Side stereo widening."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def stereo_audio(self, sample_rate):
        """1 second of stereo sine wave."""
        return create_test_tone(440.0, 1.0, sample_rate, channels=2)

    @pytest.fixture
    def mono_audio(self, sample_rate):
        """1 second of mono sine wave."""
        return create_test_tone(440.0, 1.0, sample_rate, channels=1)

    def test_output_shape_matches_stereo_input(self, stereo_audio):
        """Output shape should match stereo input."""
        output = apply_stereo_widening(stereo_audio, width=1.5)
        assert output.shape == stereo_audio.shape

    def test_mono_input_returned_unchanged(self, mono_audio):
        """Mono input should be returned unchanged."""
        output = apply_stereo_widening(mono_audio, width=1.5)
        np.testing.assert_array_equal(output, mono_audio)

    def test_width_zero_produces_mono_compatible(self, stereo_audio):
        """Width=0 should collapse to mono (L=R)."""
        output = apply_stereo_widening(stereo_audio, width=0.0)

        # Left and right channels should be identical
        np.testing.assert_array_almost_equal(output[:, 0], output[:, 1])

    def test_width_one_returns_original(self, stereo_audio):
        """Width=1 should return original stereo image."""
        output = apply_stereo_widening(stereo_audio, width=1.0)
        np.testing.assert_array_almost_equal(output, stereo_audio)

    def test_width_greater_than_one_increases_separation(self):
        """Width > 1 should increase stereo separation."""
        # Create stereo with distinct L/R content
        left = np.sin(np.linspace(0, 100, 1000))
        right = np.cos(np.linspace(0, 100, 1000))
        stereo = np.column_stack([left, right]).astype(np.float32)

        output = apply_stereo_widening(stereo, width=2.0)

        # Side channel energy should be increased
        orig_side = np.mean(np.abs(stereo[:, 0] - stereo[:, 1]))
        new_side = np.mean(np.abs(output[:, 0] - output[:, 1]))
        assert new_side > orig_side

    def test_preserves_energy_approximately(self, stereo_audio):
        """RMS energy should be preserved within tolerance."""
        output = apply_stereo_widening(stereo_audio, width=1.2)

        orig_rms = np.sqrt(np.mean(stereo_audio**2))
        new_rms = np.sqrt(np.mean(output**2))

        # Should be within 10% of original
        assert abs(new_rms - orig_rms) / orig_rms < 0.1
