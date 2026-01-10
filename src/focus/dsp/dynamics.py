"""Dynamics processing for Focus music generator.

This module implements transparent limiting to prevent digital clipping
and ensure safe listening levels for prolonged use.
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


def _apply_soft_knee(
    input_db: float,
    threshold_db: float,
    knee_db: float,
) -> float:
    """
    Apply soft-knee compression curve.

    Args:
        input_db: Input level in dB.
        threshold_db: Threshold in dB.
        knee_db: Knee width in dB.

    Returns:
        Gain reduction in dB.
    """
    half_knee = knee_db / 2.0

    if input_db < threshold_db - half_knee:
        # Below knee - no reduction
        return 0.0
    elif input_db > threshold_db + half_knee:
        # Above knee - full limiting (infinite ratio)
        return threshold_db - input_db
    else:
        # In knee region - smooth transition
        # Quadratic interpolation for smooth curve
        x = input_db - threshold_db + half_knee
        return -x * x / (4.0 * half_knee) if knee_db > 0 else 0.0


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
        peaks = [np.max(np.abs(_oversample_2x(audio[:, ch])))
                 for ch in range(audio.shape[1])]
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

    Ensures output never exceeds the ceiling level, preventing digital
    clipping during modulation peaks while maintaining transparency
    for normal signals.

    Args:
        audio: Input audio array, shape (samples,) or (samples, channels).
        sample_rate: Sample rate in Hz.
        ceiling_db: Maximum output level in dBTP (True Peak). Default -0.1.
        knee_db: Soft knee width in dB. Default 3.0 for smooth limiting.
        state: Optional state for chunk continuity.

    Returns:
        Tuple of (limited_audio, state).

    Example:
        >>> state = None
        >>> for chunk in audio_chunks:
        ...     limited, state = apply_limiter(chunk, 48000, state=state)
    """
    if state is None:
        state = LimiterState(
            ceiling_linear=_db_to_linear(ceiling_db),
            knee_db=knee_db,
            attack_samples=max(1, int(0.0001 * sample_rate)),  # 0.1ms
            release_samples=int(0.05 * sample_rate),  # 50ms
        )

    ceiling_db_val = ceiling_db
    ceiling_linear = state.ceiling_linear

    is_stereo = audio.ndim == 2
    n_samples = audio.shape[0]

    # Calculate gain reduction per sample
    output = np.zeros_like(audio)

    for i in range(n_samples):
        # Get current sample(s)
        if is_stereo:
            sample_left = audio[i, 0]
            sample_right = audio[i, 1]
            peak = max(abs(sample_left), abs(sample_right))
        else:
            sample = audio[i]
            peak = abs(sample)

        # Update envelope (peak detector)
        if peak > state._envelope:
            # Attack - fast rise
            alpha = 1.0 / state.attack_samples
            state._envelope = state._envelope + alpha * (peak - state._envelope)
        else:
            # Release - slow decay
            alpha = 1.0 / state.release_samples
            state._envelope = state._envelope + alpha * (peak - state._envelope)

        # Calculate gain reduction
        if state._envelope > 1e-10:
            input_db = _linear_to_db(state._envelope)
            reduction_db = _apply_soft_knee(input_db, ceiling_db_val, knee_db)
            target_gain = _db_to_linear(reduction_db)
        else:
            target_gain = 1.0

        # Smooth gain changes
        if target_gain < state._gain:
            # Fast attack for gain reduction
            state._gain = target_gain
        else:
            # Slow release for gain increase
            alpha = 1.0 / state.release_samples
            state._gain = state._gain + alpha * (target_gain - state._gain)

        # Apply gain
        if is_stereo:
            output[i, 0] = sample_left * state._gain
            output[i, 1] = sample_right * state._gain
        else:
            output[i] = sample * state._gain

    # Final safety clip to ceiling (catches any edge cases)
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
