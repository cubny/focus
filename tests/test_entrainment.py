"""Tests for the neural entrainment DSP module."""

import numpy as np
import pytest
from scipy import fft

from focus.dsp.entrainment import (
    ModulationState,
    apply_entrainment,
    create_test_tone,
)


class TestModulationState:
    """Tests for ModulationState phase tracking."""

    def test_initial_phase_is_zero(self):
        state = ModulationState()
        assert state.phase == 0.0

    def test_advance_updates_phase(self):
        state = ModulationState()
        # At 15 Hz and 48000 Hz sample rate, 48000 samples = 1 second = 15 cycles
        state.advance(48000, freq=15.0, sample_rate=48000)
        # Phase should wrap to 0 (15 full cycles = 30π, mod 2π = 0)
        assert abs(state.phase) < 1e-10 or abs(state.phase - 2 * np.pi) < 1e-10

    def test_phase_wraps_correctly(self):
        state = ModulationState()
        # Advance by fractional cycle
        state.advance(3200, freq=15.0, sample_rate=48000)  # 1 cycle
        expected = 2.0 * np.pi * 15.0 * 3200 / 48000
        assert abs(state.phase - (expected % (2 * np.pi))) < 1e-10


class TestApplyEntrainment:
    """Tests for the main entrainment function."""

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
        output, _ = apply_entrainment(mono_audio, sample_rate)
        assert output.shape == mono_audio.shape

    def test_output_shape_matches_input_stereo(self, stereo_audio, sample_rate):
        output, _ = apply_entrainment(stereo_audio, sample_rate)
        assert output.shape == stereo_audio.shape

    def test_output_dtype_matches_input(self, mono_audio, sample_rate):
        output, _ = apply_entrainment(mono_audio, sample_rate)
        assert output.dtype == mono_audio.dtype

    def test_zero_depth_no_change(self, mono_audio, sample_rate):
        output, _ = apply_entrainment(mono_audio, sample_rate, depth=0.0)
        np.testing.assert_array_almost_equal(output, mono_audio)

    def test_output_amplitude_reduced(self, mono_audio, sample_rate):
        """With depth > 0, output should never exceed input amplitude."""
        output, _ = apply_entrainment(mono_audio, sample_rate, depth=0.5)
        assert np.max(np.abs(output)) <= np.max(np.abs(mono_audio)) + 1e-6

    def test_returns_state_for_continuity(self, mono_audio, sample_rate):
        _, state = apply_entrainment(mono_audio, sample_rate)
        assert isinstance(state, ModulationState)
        assert state.phase != 0.0  # Phase should have advanced

    def test_phase_continuity_between_chunks(self, sample_rate):
        """Verify no discontinuity when processing consecutive chunks."""
        chunks = [create_test_tone(440.0, 0.1, sample_rate, channels=1) for _ in range(10)]

        state = ModulationState()
        outputs = []
        for chunk in chunks:
            output, state = apply_entrainment(
                chunk, sample_rate, target_freq=15.0, depth=0.3, state=state
            )
            outputs.append(output)

        # Concatenate and check for discontinuities
        full_output = np.concatenate(outputs)

        # Calculate differences between consecutive samples
        diffs = np.abs(np.diff(full_output))

        # Maximum diff should be reasonable (no clicks)
        # For a 440 Hz tone modulated at 15 Hz, expect smooth transitions
        assert np.max(diffs) < 0.1  # Threshold for detecting clicks

    def test_modulation_frequency_in_spectrum(self, sample_rate):
        """Verify modulation frequency appears in the spectrum via FFT."""
        # Create a simple DC signal for cleaner spectrum
        duration = 2.0
        audio = 0.5 * np.ones(int(duration * sample_rate), dtype=np.float32)

        target_freq = 15.0
        output, _ = apply_entrainment(audio, sample_rate, target_freq=target_freq, depth=0.5)

        # Compute FFT
        spectrum = np.abs(fft.rfft(output))
        freqs = fft.rfftfreq(len(output), 1 / sample_rate)

        # Find peak near target frequency (should be at 15 Hz)
        target_idx = np.argmin(np.abs(freqs - target_freq))
        peak_region = spectrum[target_idx - 2 : target_idx + 3]

        # DC component should be present
        assert spectrum[0] > 0

        # Should have energy at modulation frequency
        assert np.max(peak_region) > spectrum[0] * 0.01


class TestCreateTestTone:
    """Tests for the test tone generator."""

    def test_mono_shape(self):
        tone = create_test_tone(440.0, 1.0, 48000, channels=1)
        assert tone.shape == (48000,)

    def test_stereo_shape(self):
        tone = create_test_tone(440.0, 1.0, 48000, channels=2)
        assert tone.shape == (48000, 2)

    def test_duration(self):
        tone = create_test_tone(440.0, 2.5, 48000, channels=1)
        assert tone.shape == (int(2.5 * 48000),)

    def test_amplitude(self):
        tone = create_test_tone(440.0, 1.0, 48000, channels=1)
        assert np.max(np.abs(tone)) <= 0.5 + 1e-6

    def test_dtype(self):
        tone = create_test_tone(440.0, 1.0, 48000, channels=1)
        assert tone.dtype == np.float32
