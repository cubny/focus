"""FFT analysis for verifying neural entrainment modulation."""

from dataclasses import dataclass

import numpy as np
from scipy import fft


@dataclass
class ModulationAnalysis:
    """Results of modulation analysis."""

    detected_freq: float  # Hz
    modulation_index: float  # 0.0-1.0
    signal_to_noise: float  # dB
    is_valid: bool  # True if modulation detected at expected frequency


def analyze_modulation(
    audio: np.ndarray,
    sample_rate: int,
    expected_freq: float,
    tolerance: float = 1.0,
) -> ModulationAnalysis:
    """Analyze audio to detect and measure amplitude modulation.

    Uses FFT to find modulation sidebands and calculate the modulation index.

    Args:
        audio: Audio signal, mono or stereo (will be mixed to mono).
        sample_rate: Sample rate in Hz.
        expected_freq: Expected modulation frequency in Hz.
        tolerance: Frequency tolerance for peak detection in Hz.

    Returns:
        ModulationAnalysis with detected parameters.
    """
    # Mix to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Compute envelope using Hilbert transform
    analytic = np.abs(_hilbert(audio))

    # Remove DC component of envelope
    envelope = analytic - np.mean(analytic)

    # Compute FFT of envelope to find modulation frequency
    n = len(envelope)
    spectrum = np.abs(fft.rfft(envelope))
    freqs = fft.rfftfreq(n, 1 / sample_rate)

    # Find peak in the expected frequency range
    freq_mask = (freqs >= expected_freq - tolerance) & (freqs <= expected_freq + tolerance)
    if not np.any(freq_mask):
        return ModulationAnalysis(
            detected_freq=0.0, modulation_index=0.0, signal_to_noise=0.0, is_valid=False
        )

    peak_idx = np.argmax(spectrum * freq_mask)
    detected_freq = freqs[peak_idx]
    peak_power = spectrum[peak_idx]

    # Calculate noise floor (excluding peak region)
    noise_mask = ~freq_mask & (freqs > 5)  # Exclude DC and very low frequencies
    noise_floor = np.mean(spectrum[noise_mask]) if np.any(noise_mask) else 1e-10

    # Signal to noise ratio
    snr = 20 * np.log10(peak_power / max(noise_floor, 1e-10))

    # Estimate modulation index from envelope variation
    modulation_index = np.std(analytic) / (np.mean(analytic) + 1e-10)

    # Validate detection
    is_valid = abs(detected_freq - expected_freq) < tolerance and snr > 3.0

    return ModulationAnalysis(
        detected_freq=detected_freq,
        modulation_index=min(modulation_index, 1.0),
        signal_to_noise=snr,
        is_valid=is_valid,
    )


def _hilbert(x: np.ndarray) -> np.ndarray:
    """Compute the analytic signal using Hilbert transform."""
    n = len(x)
    X = fft.fft(x)

    # Create frequency response of Hilbert transformer
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2

    return fft.ifft(X * h)


def generate_report(
    audio: np.ndarray,
    sample_rate: int,
    expected_freq: float,
    expected_depth: float,
) -> str:
    """Generate a human-readable analysis report.

    Args:
        audio: Audio to analyze.
        sample_rate: Sample rate in Hz.
        expected_freq: Expected modulation frequency.
        expected_depth: Expected modulation depth (0.0-1.0).

    Returns:
        Formatted report string.
    """
    result = analyze_modulation(audio, sample_rate, expected_freq)

    lines = [
        "=" * 50,
        "Neural Entrainment Modulation Analysis",
        "=" * 50,
        f"Expected frequency: {expected_freq:.1f} Hz",
        f"Detected frequency: {result.detected_freq:.2f} Hz",
        f"Expected depth: {expected_depth:.0%}",
        f"Measured modulation index: {result.modulation_index:.1%}",
        f"Signal-to-noise ratio: {result.signal_to_noise:.1f} dB",
        "-" * 50,
    ]

    if result.is_valid:
        lines.append("✓ PASS: Modulation detected at target frequency")
    else:
        lines.append("✗ FAIL: Modulation not detected or out of range")

    lines.append("=" * 50)

    return "\n".join(lines)
