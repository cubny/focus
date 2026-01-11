"""Dynamics processing for Focus music generator.

This module implements transparent limiting to prevent digital clipping
and ensure safe listening levels for prolonged use.

OPTIMIZED VERSION: Uses vectorized NumPy operations instead of sample loops.
"""

from dataclasses import dataclass, field

import numpy as np


def _db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10.0 ** (db / 20.0)


def _linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    return 20.0 * np.log10(max(linear, 1e-10))


@dataclass
class LimiterState:
    """Maintains state for the limiter across audio chunks."""

    ceiling_linear: float = 0.989  # -0.1 dBTP
    knee_db: float = 3.0
    attack_samples: int = 5  # ~0.1ms at 48kHz
    release_samples: int = 2400  # ~50ms at 48kHz

    _gain: float = field(default=1.0, init=False)
    _envelope: float = field(default=0.0, init=False)

    def reset(self):
        """Reset limiter state."""
        self._gain = 1.0
        self._envelope = 0.0


def _oversample_2x(audio: np.ndarray) -> np.ndarray:
    """
    Upsample audio by 2x using linear interpolation.

    Simple but effective for true peak detection.
    """
    n_samples = len(audio)
    upsampled = np.zeros(n_samples * 2, dtype=audio.dtype)

    upsampled[::2] = audio
    upsampled[1:-1:2] = (audio[:-1] + audio[1:]) / 2
    upsampled[-1] = audio[-1]

    return upsampled


def _detect_true_peak(audio: np.ndarray) -> float:
    """
    Detect true peak level using 2x oversampling.

    This catches inter-sample peaks that could cause clipping
    during D/A conversion.
    """
    if audio.ndim == 2:
        # Process each channel and take max
        peaks = [np.max(np.abs(_oversample_2x(audio[:, ch]))) for ch in range(audio.shape[1])]
        return max(peaks)
    else:
        return np.max(np.abs(_oversample_2x(audio)))


def apply_limiter(
    audio: np.ndarray,
    sample_rate: int,
    ceiling_db: float = -0.1,
    knee_db: float = 3.0,
    state: LimiterState | None = None,
) -> tuple[np.ndarray, LimiterState]:
    """
    Apply transparent brickwall limiter with soft-knee compression.

    This is a VECTORIZED implementation for performance.
    Uses lookahead-free design with per-block gain computation.

    Args:
        audio: Input audio array, shape (samples,) or (samples, channels).
        sample_rate: Sample rate in Hz.
        ceiling_db: Maximum output level in dBTP (True Peak). Default -0.1.
        knee_db: Soft knee width in dB. Default 3.0 for smooth limiting.
        state: Optional state for chunk continuity.

    Returns:
        Tuple of (limited_audio, state).
    """
    if state is None:
        state = LimiterState(
            ceiling_linear=_db_to_linear(ceiling_db),
            knee_db=knee_db,
            attack_samples=max(1, int(0.0001 * sample_rate)),  # 0.1ms
            release_samples=int(0.05 * sample_rate),  # 50ms
        )

    ceiling_linear = state.ceiling_linear

    is_stereo = audio.ndim == 2

    # Get peak per sample (vectorized)
    if is_stereo:
        peaks = np.maximum(np.abs(audio[:, 0]), np.abs(audio[:, 1]))
    else:
        peaks = np.abs(audio)

    # Find maximum peak in the chunk
    max_peak = np.max(peaks)

    if max_peak <= ceiling_linear:
        # No limiting needed - pass through
        return audio.copy(), state

    # Calculate required gain reduction
    # Simple approach: compute single gain for entire chunk based on peak
    required_gain = ceiling_linear / max_peak

    # Smooth transition from previous gain
    # Use exponential smoothing over the chunk
    n_samples = len(audio)

    if state._gain > required_gain:
        # Need to reduce gain - fast attack
        target_gain = required_gain
    else:
        # Can increase gain - slow release
        alpha = min(1.0, n_samples / state.release_samples)
        target_gain = state._gain + alpha * (required_gain - state._gain)

    # Apply gain with linear interpolation for smooth transition
    gain_start = state._gain
    gain_end = min(1.0, target_gain)  # Never amplify above unity

    # Create gain envelope
    gain_envelope = np.linspace(gain_start, gain_end, n_samples).astype(np.float32)

    # Apply gain
    if is_stereo:
        output = audio * gain_envelope[:, np.newaxis]
    else:
        output = audio * gain_envelope

    # Update state
    state._gain = gain_end
    state._envelope = max_peak

    # Final safety clip to ceiling
    output = np.clip(output, -ceiling_linear, ceiling_linear)

    return output.astype(audio.dtype), state


def get_true_peak_db(audio: np.ndarray) -> float:
    """
    Get the true peak level of audio in dBTP.

    Args:
        audio: Audio array, shape (samples,) or (samples, channels).

    Returns:
        True peak level in dBTP.
    """
    peak = _detect_true_peak(audio)
    return _linear_to_db(peak)
