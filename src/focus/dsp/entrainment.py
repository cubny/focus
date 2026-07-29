"""Neural entrainment via amplitude modulation.

This module implements the core DSP algorithm for inducing neural phase locking
through rapid amplitude modulation in the Beta frequency range (12-20 Hz).
"""

from dataclasses import dataclass, field

import numpy as np

# scipy is a core dependency, but guard it so the DSP degrades gracefully to
# full-spectrum modulation if it is ever missing (mirrors the numpy/genai
# availability-flag convention used elsewhere in the package).
try:
    from scipy.signal import butter, lfilter

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Order of the Butterworth low-pass used to isolate the modulation band.
_BAND_FILTER_ORDER = 2


@dataclass
class ModulationState:
    """Maintains phase (and band-split filter) continuity between audio chunks."""

    phase: float = 0.0

    # Low-pass filter state for band-limited modulation. These are threaded back
    # into each call so the band split is click-free across chunk boundaries.
    _lp_b: np.ndarray | None = field(default=None, repr=False)
    _lp_a: np.ndarray | None = field(default=None, repr=False)
    _lp_zi: np.ndarray | None = field(default=None, repr=False)
    _lp_cutoff: float | None = field(default=None, repr=False)
    _lp_sr: int | None = field(default=None, repr=False)
    _lp_channels: int | None = field(default=None, repr=False)

    def advance(self, samples: int, freq: float, sample_rate: int) -> None:
        """Advance phase by the given number of samples."""
        self.phase += 2.0 * np.pi * freq * samples / sample_rate
        # Keep phase in [0, 2π) to prevent float precision issues
        self.phase = self.phase % (2.0 * np.pi)


def _lowpass_band(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float, state: ModulationState
) -> np.ndarray:
    """Return the low-frequency band of ``audio`` using a stateful Butterworth filter.

    Filter coefficients and delay state live on ``state`` so the split stays
    continuous (no clicks) across chunk boundaries.
    """
    channels = audio.shape[1] if audio.ndim == 2 else 1

    needs_init = (
        state._lp_zi is None
        or state._lp_cutoff != cutoff_hz
        or state._lp_sr != sample_rate
        or state._lp_channels != channels
    )
    if needs_init:
        nyquist = 0.5 * sample_rate
        wn = min(max(cutoff_hz / nyquist, 1e-4), 0.99)
        b, a = butter(_BAND_FILTER_ORDER, wn, btype="low")
        state._lp_b = b
        state._lp_a = a
        state._lp_cutoff = cutoff_hz
        state._lp_sr = sample_rate
        state._lp_channels = channels
        zi_len = max(len(a), len(b)) - 1
        if audio.ndim == 2:
            state._lp_zi = np.zeros((zi_len, channels), dtype=np.float64)
        else:
            state._lp_zi = np.zeros(zi_len, dtype=np.float64)

    low, state._lp_zi = lfilter(state._lp_b, state._lp_a, audio, axis=0, zi=state._lp_zi)
    return low


def apply_entrainment(
    audio: np.ndarray,
    sample_rate: int,
    target_freq: float = 15.0,
    depth: float = 0.15,
    state: ModulationState | None = None,
    band_cutoff_hz: float | None = 500.0,
) -> tuple[np.ndarray, ModulationState]:
    """
    Apply amplitude modulation for neural entrainment.

    By default the modulation is *band-limited*: only the low-frequency band
    (below ``band_cutoff_hz``) is amplitude-modulated, while the mids/highs that
    carry the perceived melody pass through untouched. This keeps the
    entrainment pulse working in the background (felt as gentle rhythmic energy)
    without the whole mix audibly "throbbing" as a tremolo.

    Args:
        audio: Input audio array, shape (samples,) for mono or (samples, channels) for stereo.
               Expected dtype is float32 or float64 with values in [-1, 1].
        sample_rate: Sample rate of the audio in Hz.
        target_freq: Modulation frequency in Hz. Use 12-20 Hz for Beta wave entrainment.
                     Higher frequencies (18-20 Hz) for intense focus, lower (12-14 Hz) for
                     light concentration.
        depth: Modulation depth from 0.0 (no effect) to 1.0 (full modulation).
               Recommended range is ~0.1-0.2 for a subtle, non-distracting effect.
        state: Optional state object for phase (and filter) continuity between chunks.
               Pass the returned state to subsequent calls to prevent clicks.
        band_cutoff_hz: Upper edge of the modulated low band in Hz. Set to ``None``
                        (or if scipy is unavailable) to modulate the full spectrum.

    Returns:
        Tuple of (modulated_audio, state). The state should be passed to the next
        call to maintain phase continuity.

    Example:
        >>> state = ModulationState()
        >>> for chunk in audio_chunks:
        ...     modulated, state = apply_entrainment(chunk, 48000, 15.0, 0.15, state)
        ...     play(modulated)
    """
    if state is None:
        state = ModulationState()

    n_samples = audio.shape[0]

    # Generate time array starting from current phase
    t = np.arange(n_samples) / sample_rate
    phase_array = 2.0 * np.pi * target_freq * t + state.phase

    # Create modulation envelope: oscillates between (1-depth) and 1.0.
    # (1 - depth) + depth * (0.5 * (1 + cos(...))) gives range [1-depth, 1],
    # so we only ever reduce volume, never amplify.
    modulator = (1.0 - depth) + depth * (0.5 * (1.0 + np.cos(phase_array)))

    # Reshape modulator for stereo audio
    if audio.ndim == 2:
        modulator = modulator[:, np.newaxis]

    use_band = band_cutoff_hz is not None and SCIPY_AVAILABLE and depth > 0.0 and n_samples > 0
    if use_band:
        # Split into the low modulation band and the untouched remainder, then
        # modulate only the low band and recombine.
        low = _lowpass_band(audio, sample_rate, band_cutoff_hz, state)
        rest = audio - low
        modulated = low * modulator + rest
    else:
        # Full-spectrum modulation (fallback when band-limiting is disabled).
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
