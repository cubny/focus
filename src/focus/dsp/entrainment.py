"""Neural entrainment via amplitude modulation.

This module implements the core DSP algorithm for inducing neural phase locking
through rapid amplitude modulation in the Beta frequency range (12-20 Hz).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ModulationState:
    """Maintains phase continuity between audio chunks."""

    phase: float = 0.0

    def advance(self, samples: int, freq: float, sample_rate: int) -> None:
        """Advance phase by the given number of samples."""
        self.phase += 2.0 * np.pi * freq * samples / sample_rate
        # Keep phase in [0, 2π) to prevent float precision issues
        self.phase = self.phase % (2.0 * np.pi)


def apply_entrainment(
    audio: np.ndarray,
    sample_rate: int,
    target_freq: float = 15.0,
    depth: float = 0.3,
    state: ModulationState | None = None,
) -> tuple[np.ndarray, ModulationState]:
    """
    Apply amplitude modulation for neural entrainment.

    The modulation creates a subtle "tremolo" effect that oscillates at the
    target frequency, inducing neural phase locking in the listener.

    Args:
        audio: Input audio array, shape (samples,) for mono or (samples, channels) for stereo.
               Expected dtype is float32 or float64 with values in [-1, 1].
        sample_rate: Sample rate of the audio in Hz.
        target_freq: Modulation frequency in Hz. Use 12-20 Hz for Beta wave entrainment.
                     Higher frequencies (18-20 Hz) for intense focus, lower (12-14 Hz) for
                     light concentration.
        depth: Modulation depth from 0.0 (no effect) to 1.0 (full modulation).
               Recommended range is 0.2-0.4 for noticeable but non-distracting effect.
        state: Optional state object for phase continuity between chunks.
               Pass the returned state to subsequent calls to prevent clicks.

    Returns:
        Tuple of (modulated_audio, state). The state should be passed to the next
        call to maintain phase continuity.

    Example:
        >>> state = ModulationState()
        >>> for chunk in audio_chunks:
        ...     modulated, state = apply_entrainment(chunk, 48000, 15.0, 0.3, state)
        ...     play(modulated)
    """
    if state is None:
        state = ModulationState()

    n_samples = audio.shape[0]

    # Generate time array starting from current phase
    t = np.arange(n_samples) / sample_rate
    phase_array = 2.0 * np.pi * target_freq * t + state.phase

    # Create modulation envelope: oscillates between (1-depth) and 1.0
    # Using (1 - depth/2) + (depth/2) * cos(...) gives range [1-depth, 1]
    # This ensures we only reduce volume, never amplify
    modulator = (1.0 - depth) + depth * (0.5 * (1.0 + np.cos(phase_array)))

    # Reshape modulator for stereo audio
    if audio.ndim == 2:
        modulator = modulator[:, np.newaxis]

    # Apply modulation
    modulated = audio * modulator

    # Update state for next chunk
    state.advance(n_samples, target_freq, sample_rate)

    return modulated.astype(audio.dtype), state


def apply_fade_in(
    audio: np.ndarray,
    sample_rate: int,
    duration: float = 0.5,
) -> np.ndarray:
    """
    Apply a fade-in envelope to the start of audio.

    Args:
        audio: Input audio array, shape (samples,) for mono or (samples, channels) for stereo.
        sample_rate: Sample rate of the audio in Hz.
        duration: Fade duration in seconds.

    Returns:
        Audio with fade-in applied.
    """
    fade_samples = int(duration * sample_rate)
    fade_samples = min(fade_samples, audio.shape[0])

    if fade_samples == 0:
        return audio

    # Create smooth fade envelope using cosine curve (0 -> 1)
    t = np.linspace(0, np.pi / 2, fade_samples)
    envelope = np.sin(t) ** 2  # Smooth S-curve for natural fade

    # Apply to audio
    result = audio.copy()
    if audio.ndim == 2:
        result[:fade_samples] *= envelope[:, np.newaxis]
    else:
        result[:fade_samples] *= envelope

    return result.astype(audio.dtype)


def apply_fade_out(
    audio: np.ndarray,
    sample_rate: int,
    duration: float = 2.0,
) -> np.ndarray:
    """
    Apply a fade-out envelope to the end of audio.

    Args:
        audio: Input audio array, shape (samples,) for mono or (samples, channels) for stereo.
        sample_rate: Sample rate of the audio in Hz.
        duration: Fade duration in seconds.

    Returns:
        Audio with fade-out applied.
    """
    fade_samples = int(duration * sample_rate)
    fade_samples = min(fade_samples, audio.shape[0])

    if fade_samples == 0:
        return audio

    # Create smooth fade envelope using cosine curve (1 -> 0)
    t = np.linspace(0, np.pi / 2, fade_samples)
    envelope = np.cos(t) ** 2  # Smooth S-curve for natural fade

    # Apply to audio
    result = audio.copy()
    if audio.ndim == 2:
        result[-fade_samples:] *= envelope[:, np.newaxis]
    else:
        result[-fade_samples:] *= envelope

    return result.astype(audio.dtype)


def create_test_tone(
    freq: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 48000,
    channels: int = 2,
) -> np.ndarray:
    """
    Create a test sine wave for verification.

    Args:
        freq: Frequency of the test tone in Hz.
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels (1 for mono, 2 for stereo).

    Returns:
        Audio array of shape (samples,) or (samples, channels).
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    tone = 0.5 * np.sin(2.0 * np.pi * freq * t)

    if channels == 2:
        tone = np.column_stack([tone, tone])

    return tone.astype(np.float32)
