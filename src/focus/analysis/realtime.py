"""Real-time spectrum analysis for the live terminal visualizer.

Reuses the same FFT primitives as :mod:`focus.analysis.fft` (``scipy.fft``), but
maintains a rolling window and per-band smoothing state so it can be driven at
video frame rates from the playback loop.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq


class SpectrumAnalyzer:
    """Turns a stream of audio chunks into normalized log-spaced band levels.

    All work happens on the caller's thread (the asyncio event loop); the
    PortAudio callback is never involved. ``push`` is cheap (a downmix + buffer
    roll); ``compute`` performs one ``rfft`` per call.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        fft_size: int = 2048,
        f_min: float = 30.0,
        f_max: float = 16000.0,
        floor_db: float = -70.0,
        ceil_db: float = -12.0,
        attack: float = 0.6,
        decay: float = 0.18,
    ) -> None:
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.f_min = f_min
        self.f_max = min(f_max, sample_rate / 2.0)
        self.floor_db = floor_db
        self.ceil_db = ceil_db
        self.attack = attack
        self.decay = decay

        self._buf = np.zeros(fft_size, dtype=np.float32)
        self._window = np.hanning(fft_size).astype(np.float32)
        self._freqs = rfftfreq(fft_size, 1.0 / sample_rate)
        # Coherent gain of the window: normalizing by it makes a full-scale
        # sine read ~0 dBFS, so the floor/ceil dB thresholds are meaningful and
        # independent of fft_size.
        self._norm = self._window.sum() / 2.0

        # Smoothing state and cached band edges, both keyed by num_bands.
        self._levels: np.ndarray | None = None
        self._band_bins: list[tuple[int, int]] | None = None
        self._num_bands: int | None = None

    def push(self, chunk: np.ndarray) -> None:
        """Feed a new audio chunk (mono or stereo float array)."""
        if chunk is None or len(chunk) == 0:
            return
        mono = chunk.mean(axis=1) if chunk.ndim == 2 else chunk
        mono = np.asarray(mono, dtype=np.float32)
        n = len(mono)
        if n >= self.fft_size:
            self._buf[:] = mono[-self.fft_size :]
        else:
            self._buf = np.roll(self._buf, -n)
            self._buf[-n:] = mono

    def push_silence(self) -> None:
        """Zero the rolling window so bars fall to the floor (e.g. while paused)."""
        self._buf[:] = 0.0

    def _ensure_bands(self, num_bands: int) -> None:
        if self._num_bands == num_bands and self._band_bins is not None:
            return
        edges = np.logspace(np.log10(self.f_min), np.log10(self.f_max), num_bands + 1)
        idx = np.searchsorted(self._freqs, edges)
        bins: list[tuple[int, int]] = []
        max_bin = len(self._freqs)
        for i in range(num_bands):
            lo = int(idx[i])
            hi = int(idx[i + 1])
            # Guarantee at least one bin per band so high bands aren't empty.
            if hi <= lo:
                hi = min(lo + 1, max_bin)
            bins.append((lo, hi))
        self._band_bins = bins
        self._num_bands = num_bands
        self._levels = np.zeros(num_bands, dtype=np.float32)

    def compute(self, num_bands: int) -> np.ndarray:
        """Return smoothed band levels in ``[0, 1]`` (length ``num_bands``)."""
        self._ensure_bands(num_bands)
        assert self._band_bins is not None and self._levels is not None

        mag = np.abs(rfft(self._buf * self._window)) / self._norm
        db = 20.0 * np.log10(mag + 1e-9)

        target = np.empty(num_bands, dtype=np.float32)
        span = self.ceil_db - self.floor_db
        for i, (lo, hi) in enumerate(self._band_bins):
            band_db = db[lo:hi].max() if hi > lo else self.floor_db
            target[i] = np.clip((band_db - self.floor_db) / span, 0.0, 1.0)

        # Fast attack, slow decay; never dip below the instantaneous target.
        prev = self._levels
        rising = target >= prev
        smoothed = np.where(
            rising,
            self.attack * target + (1.0 - self.attack) * prev,
            np.maximum(prev - self.decay, target),
        )
        self._levels = np.clip(smoothed, 0.0, 1.0).astype(np.float32)
        return self._levels
