"""Spatial audio processing for Focus music generator.

This module implements reverb and stereo widening effects to create
a comfortable, polished sound field for long-term listening.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CombFilter:
    """Single comb filter for reverb implementation."""

    delay_samples: int
    feedback: float = 0.7
    damping: float = 0.5

    _buffer: np.ndarray = field(default=None, init=False, repr=False)
    _buffer_idx: int = field(default=0, init=False)
    _filter_state: float = field(default=0.0, init=False)

    def __post_init__(self):
        self._buffer = np.zeros(self.delay_samples, dtype=np.float32)

    def process(self, input_sample: float) -> float:
        """Process a single sample through the comb filter."""
        delayed = self._buffer[self._buffer_idx]

        # Low-pass filter for damping (simple one-pole)
        self._filter_state = delayed * (1 - self.damping) + self._filter_state * self.damping

        # Write new sample with feedback
        self._buffer[self._buffer_idx] = input_sample + self._filter_state * self.feedback

        # Advance buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self.delay_samples

        return delayed

    def reset(self):
        """Clear the filter state."""
        self._buffer.fill(0)
        self._buffer_idx = 0
        self._filter_state = 0.0


@dataclass
class AllpassFilter:
    """Single allpass filter for reverb diffusion."""

    delay_samples: int
    feedback: float = 0.5

    _buffer: np.ndarray = field(default=None, init=False, repr=False)
    _buffer_idx: int = field(default=0, init=False)

    def __post_init__(self):
        self._buffer = np.zeros(self.delay_samples, dtype=np.float32)

    def process(self, input_sample: float) -> float:
        """Process a single sample through the allpass filter."""
        delayed = self._buffer[self._buffer_idx]

        output = -input_sample + delayed
        self._buffer[self._buffer_idx] = input_sample + delayed * self.feedback

        self._buffer_idx = (self._buffer_idx + 1) % self.delay_samples

        return output

    def reset(self):
        """Clear the filter state."""
        self._buffer.fill(0)
        self._buffer_idx = 0


@dataclass
class ReverbState:
    """Maintains state for the Schroeder reverb across audio chunks.

    Uses 4 comb filters in parallel followed by 2 allpass filters in series
    (Schroeder reverberator architecture).
    """

    sample_rate: int = 48000
    room_size: float = 0.5
    damping: float = 0.5

    # Internal filters (initialized in __post_init__)
    _comb_filters: list = field(default_factory=list, init=False, repr=False)
    _allpass_filters: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # Comb filter delay times in ms (mutually prime for density)
        comb_delays_ms = [29.7, 37.1, 41.1, 43.7]

        # Scale delays by room size (0.5 - 1.5x)
        scale = 0.5 + self.room_size

        self._comb_filters = [
            CombFilter(
                delay_samples=int(d * 0.001 * self.sample_rate * scale),
                feedback=0.84 * self.room_size + 0.1,
                damping=self.damping
            )
            for d in comb_delays_ms
        ]

        # Allpass filter delay times in ms
        allpass_delays_ms = [5.0, 1.7]

        self._allpass_filters = [
            AllpassFilter(
                delay_samples=max(1, int(d * 0.001 * self.sample_rate)),
                feedback=0.5
            )
            for d in allpass_delays_ms
        ]

    def reset(self):
        """Clear all filter states."""
        for f in self._comb_filters:
            f.reset()
        for f in self._allpass_filters:
            f.reset()


def apply_reverb(
    audio: np.ndarray,
    sample_rate: int,
    room_size: float = 0.3,
    damping: float = 0.5,
    wet_dry_mix: float = 0.15,
    state: ReverbState | None = None,
) -> tuple[np.ndarray, ReverbState]:
    """
    Apply Schroeder reverb to audio for a subtle room ambience.

    Args:
        audio: Input audio array, shape (samples,) for mono or (samples, channels) for stereo.
        sample_rate: Sample rate of the audio in Hz.
        room_size: Room size from 0.0 (small) to 1.0 (large). Default 0.3 for subtle effect.
        damping: High-frequency damping from 0.0 (bright) to 1.0 (dark). Default 0.5.
        wet_dry_mix: Mix ratio from 0.0 (dry only) to 1.0 (wet only). Default 0.15 for subtle.
        state: Optional state for chunk continuity. Pass returned state to next call.

    Returns:
        Tuple of (processed_audio, state).

    Example:
        >>> state = None
        >>> for chunk in audio_chunks:
        ...     processed, state = apply_reverb(chunk, 48000, state=state)
    """
    if state is None:
        state = ReverbState(sample_rate=sample_rate, room_size=room_size, damping=damping)

    is_stereo = audio.ndim == 2

    # Convert to mono for reverb processing (sum channels)
    if is_stereo:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    n_samples = len(mono)
    wet = np.zeros(n_samples, dtype=np.float32)

    # Process through Schroeder reverb
    for i in range(n_samples):
        sample = mono[i]

        # Parallel comb filters (sum outputs)
        comb_sum = 0.0
        for comb in state._comb_filters:
            comb_sum += comb.process(sample)
        comb_sum /= len(state._comb_filters)

        # Series allpass filters
        allpass_out = comb_sum
        for allpass in state._allpass_filters:
            allpass_out = allpass.process(allpass_out)

        wet[i] = allpass_out

    # Mix wet and dry
    dry = mono if not is_stereo else audio

    if is_stereo:
        # Apply reverb to both channels
        wet_stereo = np.column_stack([wet, wet])
        output = dry * (1 - wet_dry_mix) + wet_stereo * wet_dry_mix
    else:
        output = dry * (1 - wet_dry_mix) + wet * wet_dry_mix

    return output.astype(audio.dtype), state


def apply_stereo_widening(
    audio: np.ndarray,
    width: float = 1.2,
) -> np.ndarray:
    """
    Apply stereo widening using Mid-Side (M/S) processing.

    Args:
        audio: Stereo audio array, shape (samples, 2).
        width: Stereo width multiplier.
               0.0 = mono (L=R)
               1.0 = original stereo image
               2.0 = extra wide (exaggerated stereo)
               Default 1.2 for subtle enhancement.

    Returns:
        Stereo audio with adjusted width.

    Note:
        - Mono input is returned unchanged.
        - Values > 2.0 may cause phase issues on mono playback systems.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        # Not stereo, return unchanged
        return audio

    left = audio[:, 0]
    right = audio[:, 1]

    # Convert to Mid-Side
    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    # Apply width to side channel
    side_widened = side * width

    # Convert back to Left-Right
    left_out = mid + side_widened
    right_out = mid - side_widened

    output = np.column_stack([left_out, right_out])

    return output.astype(audio.dtype)
