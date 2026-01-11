"""Spatial audio processing for Focus music generator.

This module implements reverb and stereo widening effects to create
a comfortable, polished sound field for long-term listening.

OPTIMIZED VERSION: Uses vectorized NumPy operations with scipy for convolution.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass
class ReverbState:
    """Maintains state for the convolution reverb across audio chunks.

    Uses a simple algorithmic reverb based on all-pass and comb delay networks,
    but processed in blocks rather than sample-by-sample.
    """

    sample_rate: int = 48000
    room_size: float = 0.3
    damping: float = 0.5

    # Delay line buffers
    _delay_buffers: list = field(default_factory=list, init=False, repr=False)
    _delay_indices: list = field(default_factory=list, init=False, repr=False)
    _filter_states: list = field(default_factory=list, init=False, repr=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self):
        self._initialize_delays()

    def _initialize_delays(self):
        """Initialize delay line buffers."""
        # Delay times in samples (mutually prime for density)
        # These are typical values for a small room reverb
        scale = 0.5 + self.room_size
        delay_ms = [29.7, 37.1, 41.1, 43.7]  # Comb delays

        self._delay_buffers = []
        self._delay_indices = []
        self._filter_states = []

        for d in delay_ms:
            delay_samples = int(d * 0.001 * self.sample_rate * scale)
            self._delay_buffers.append(np.zeros(delay_samples, dtype=np.float32))
            self._delay_indices.append(0)
            self._filter_states.append(0.0)

        self._initialized = True

    def reset(self):
        """Clear all delay buffers."""
        for buf in self._delay_buffers:
            buf.fill(0)
        self._delay_indices = [0] * len(self._delay_indices)
        self._filter_states = [0.0] * len(self._filter_states)


def _process_comb_vectorized(
    audio: np.ndarray,
    delay_buffer: np.ndarray,
    delay_idx: int,
    filter_state: float,
    feedback: float,
    damping: float,
) -> tuple[np.ndarray, int, float]:
    """Process audio through a comb filter using block processing.

    This is more efficient than sample-by-sample when chunk size is reasonable.
    """
    n_samples = len(audio)
    delay_len = len(delay_buffer)
    output = np.zeros(n_samples, dtype=np.float32)

    # Process in blocks that fit within the delay buffer
    block_size = min(n_samples, delay_len)

    pos = 0
    current_filter_state = filter_state
    current_idx = delay_idx

    while pos < n_samples:
        end = min(pos + block_size, n_samples)
        block_len = end - pos

        # Read from delay buffer
        read_indices = (current_idx + np.arange(block_len)) % delay_len
        delayed = delay_buffer[read_indices]

        # Output is the delayed signal
        output[pos:end] = delayed

        # Apply feedback with damping (vectorized low-pass)
        # For simplicity, use single-pole IIR approximation per block
        damped = delayed * (1 - damping) + np.roll(delayed, 1) * damping
        damped[0] = delayed[0] * (1 - damping) + current_filter_state * damping
        current_filter_state = damped[-1]

        # Write back to delay buffer
        new_values = audio[pos:end] + damped * feedback
        delay_buffer[read_indices] = new_values

        current_idx = (current_idx + block_len) % delay_len
        pos = end

    return output, current_idx, current_filter_state


def apply_reverb(
    audio: np.ndarray,
    sample_rate: int,
    room_size: float = 0.3,
    damping: float = 0.5,
    wet_dry_mix: float = 0.15,
    state: ReverbState | None = None,
) -> tuple[np.ndarray, ReverbState]:
    """
    Apply algorithmic reverb to audio for a subtle room ambience.

    OPTIMIZED: Uses block-based processing instead of sample-by-sample.

    Args:
        audio: Input audio array, shape (samples,) for mono or (samples, channels) for stereo.
        sample_rate: Sample rate of the audio in Hz.
        room_size: Room size from 0.0 (small) to 1.0 (large). Default 0.3 for subtle effect.
        damping: High-frequency damping from 0.0 (bright) to 1.0 (dark). Default 0.5.
        wet_dry_mix: Mix ratio from 0.0 (dry only) to 1.0 (wet only). Default 0.15 for subtle.
        state: Optional state for chunk continuity. Pass returned state to next call.

    Returns:
        Tuple of (processed_audio, state).
    """
    if state is None:
        state = ReverbState(sample_rate=sample_rate, room_size=room_size, damping=damping)

    if not state._initialized:
        state._initialize_delays()

    is_stereo = audio.ndim == 2

    # Convert to mono for reverb processing
    if is_stereo:
        mono = np.mean(audio, axis=1).astype(np.float32)
    else:
        mono = audio.astype(np.float32)

    n_samples = len(mono)

    # Calculate feedback based on room size
    feedback = 0.84 * state.room_size + 0.1

    # Process through parallel comb filters and sum
    wet = np.zeros(n_samples, dtype=np.float32)

    for i, (delay_buf, delay_idx, filter_state) in enumerate(
        zip(state._delay_buffers, state._delay_indices, state._filter_states)
    ):
        comb_out, new_idx, new_filter = _process_comb_vectorized(
            mono, delay_buf, delay_idx, filter_state, feedback, state.damping
        )
        wet += comb_out
        state._delay_indices[i] = new_idx
        state._filter_states[i] = new_filter

    # Normalize by number of comb filters
    wet /= len(state._delay_buffers)

    # Simple all-pass diffusion (single pass with fixed delay)
    # This adds density without the slow sample-by-sample processing
    allpass_delay = int(0.005 * sample_rate)  # 5ms
    if allpass_delay > 0 and len(wet) > allpass_delay:
        delayed = np.concatenate([np.zeros(allpass_delay), wet[:-allpass_delay]])
        wet = 0.5 * (-wet + delayed + wet * 0.5)

    # Mix wet and dry
    if is_stereo:
        # Apply wet signal to both channels
        wet_stereo = np.column_stack([wet, wet])
        output = audio * (1 - wet_dry_mix) + wet_stereo * wet_dry_mix
    else:
        output = audio * (1 - wet_dry_mix) + wet * wet_dry_mix

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
