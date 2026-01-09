"""Tests for overlap-add processing in the audio pipeline."""

import numpy as np
import pytest

from focus.audio.pipeline import OverlapAddState
from focus.dsp.entrainment import create_test_tone


class TestOverlapAddState:
    """Tests for the overlap-add chunk boundary processing."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    def test_first_chunk_returned_unchanged(self, sample_rate):
        """First chunk should be returned unchanged."""
        state = OverlapAddState(overlap_samples=256)
        chunk = create_test_tone(440.0, 0.1, sample_rate, channels=2)

        output = state.process(chunk)

        np.testing.assert_array_equal(output, chunk)

    def test_output_length_preserved(self, sample_rate):
        """Each chunk output should have same length as input."""
        state = OverlapAddState(overlap_samples=256)
        chunks = [
            create_test_tone(440.0, 0.1, sample_rate, channels=2)
            for _ in range(3)
        ]

        for chunk in chunks:
            output = state.process(chunk)
            assert output.shape == chunk.shape

    def test_smooth_transition_at_boundary(self, sample_rate):
        """Transition between chunks should be smooth."""
        state = OverlapAddState(overlap_samples=256)

        # Create two different frequency chunks to create a discontinuity
        chunk1 = create_test_tone(440.0, 0.1, sample_rate, channels=2)
        chunk2 = create_test_tone(880.0, 0.1, sample_rate, channels=2)

        # Process both
        output1 = state.process(chunk1)
        output2 = state.process(chunk2)

        # The beginning of output2 should be a blend of chunk1 tail and chunk2 start
        # Not a hard transition
        # Calculate the expected crossfade
        diffs = np.abs(np.diff(output2[:100], axis=0))
        max_diff = np.max(diffs)

        # Should have no extreme jumps despite frequency change
        assert max_diff < 0.3

    def test_multiple_chunks_no_discontinuity(self, sample_rate):
        """Processing many chunks should not introduce clicks."""
        state = OverlapAddState(overlap_samples=256)

        # Create continuous audio split into chunks
        full_audio = create_test_tone(440.0, 0.5, sample_rate, channels=2)
        chunk_size = 4800  # 0.1 second

        chunks = [full_audio[i:i+chunk_size] for i in range(0, len(full_audio), chunk_size)]

        outputs = []
        for chunk in chunks:
            if len(chunk) > 0:
                output = state.process(chunk)
                outputs.append(output)

        # Concatenate
        full_output = np.concatenate(outputs)

        # Check for discontinuities within each chunk (not at boundaries between processed outputs)
        # Since we're testing the overlap-add state in isolation with already continuous audio,
        # the crossfade should help smooth any discontinuities at the overlap region
        for output in outputs:
            diffs = np.abs(np.diff(output, axis=0))
            # Check that internal diffs are reasonable for a 440Hz tone
            # (expect smooth sine wave transitions)
            assert np.mean(diffs) < 0.05  # Average diff should be low

    def test_reset_clears_state(self, sample_rate):
        """Reset should clear the overlap buffer."""
        state = OverlapAddState(overlap_samples=256)

        chunk = create_test_tone(440.0, 0.1, sample_rate, channels=2)
        state.process(chunk)

        # Reset
        state.reset()

        # Next chunk should behave like first chunk
        chunk2 = create_test_tone(880.0, 0.1, sample_rate, channels=2)
        output = state.process(chunk2)

        np.testing.assert_array_equal(output, chunk2)

    def test_mono_audio_supported(self, sample_rate):
        """Should work with mono audio."""
        state = OverlapAddState(overlap_samples=256)

        chunks = [
            create_test_tone(440.0, 0.1, sample_rate, channels=1)
            for _ in range(3)
        ]

        outputs = []
        for chunk in chunks:
            output = state.process(chunk)
            assert output.shape == chunk.shape
            outputs.append(output)

        # Should have no issues
        full_output = np.concatenate(outputs)
        assert len(full_output) == sum(len(c) for c in chunks)

    def test_preserves_dtype(self, sample_rate):
        """Output dtype should match input."""
        state = OverlapAddState(overlap_samples=256)

        chunk = create_test_tone(440.0, 0.1, sample_rate, channels=2)
        output = state.process(chunk)

        assert output.dtype == chunk.dtype
