"""Audio output using sounddevice for real-time playback.

This module provides robust real-time audio output with:
- Ring buffer for smooth playback
- Automatic recovery from underruns with fade-in
- Thread-safe operations
"""

import queue
from dataclasses import dataclass, field

import numpy as np

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    sd = None
    SOUNDDEVICE_AVAILABLE = False


@dataclass
class AudioOutput:
    """Real-time audio output using sounddevice with robust underrun handling.

    Key features:
    - Large ring buffer to absorb timing jitter from audio sources
    - Deferred stream start - waits for buffer to fill before playing
    - Automatic fade-in recovery after underruns (no clicks)
    - Graceful fade-out during underruns
    - Thread-safe queue operations
    """

    sample_rate: int = 48000
    channels: int = 2
    blocksize: int = 2048  # ~42ms per block at 48kHz
    buffersize: int = 200  # Queue capacity in blocks (~8.5 seconds at 48kHz)
    minimum_buffer_seconds: float = 3.0  # Pre-fill buffer before starting playback
    volume: float = 1.0  # Output gain, 0.0-1.0 (attenuation only)

    _stream: object = field(default=None, init=False, repr=False)
    _queue: queue.Queue = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)
    _paused: bool = field(default=False, init=False)
    _stream_active: bool = field(default=False, init=False)
    _leftover: np.ndarray | None = field(default=None, init=False, repr=False)
    _underrun_count: int = field(default=0, init=False)

    # Underrun recovery state
    _last_good_block: np.ndarray | None = field(default=None, init=False, repr=False)
    _underrun_fade_pos: int = field(default=0, init=False)  # Fade-out position during underrun
    _recovery_fade_pos: int = field(default=0, init=False)  # Fade-in position after underrun
    _in_underrun: bool = field(default=False, init=False)
    _recovery_samples: int = field(default=0, init=False)  # Samples to fade in over

    # Statistics
    _blocks_written: int = field(default=0, init=False)
    _blocks_played: int = field(default=0, init=False)

    def __post_init__(self):
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice package is required. Install with: pip install sounddevice"
            )
        self._queue = queue.Queue(maxsize=self.buffersize)
        # Recovery fade-in over ~50ms for smooth transitions
        self._recovery_samples = int(0.05 * self.sample_rate)

    def start(self) -> None:
        """Prepare for audio output (stream starts when buffer is filled)."""
        self._running = True
        self._paused = False
        self._stream_active = False
        self._in_underrun = False
        self._underrun_fade_pos = 0
        self._recovery_fade_pos = 0
        self._started = True
        # Note: Stream is NOT started here - it starts when buffer is pre-filled

    def _maybe_start_stream(self) -> None:
        """Start the actual audio stream if we have enough buffer pre-filled."""
        if self._stream_active or not self._running or self._paused:
            return

        # Check if we have enough buffer
        if self.buffer_seconds >= self.minimum_buffer_seconds:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.blocksize,
                callback=self._audio_callback,
                finished_callback=self._stream_finished,
            )
            self._stream.start()
            self._stream_active = True

    def _stream_finished(self) -> None:
        """Called when stream finishes (for debugging)."""
        pass

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback for sounddevice to get audio data.

        Implements robust underrun handling:
        1. Normal operation: read from queue, reset recovery state
        2. Underrun: fade out gracefully over multiple blocks
        3. Recovery: when data returns, fade in smoothly (no click)
        """
        if status and status.output_underflow:
            self._underrun_count += 1

        try:
            data = self._queue.get_nowait()
            self._blocks_played += 1

            # Check if we're recovering from an underrun
            if self._in_underrun:
                # Apply fade-in to prevent click when audio returns
                fade_in_samples = min(self._recovery_samples, frames)
                fade_envelope = np.linspace(0.0, 1.0, fade_in_samples)

                # Apply fade to beginning of block
                if data.ndim == 2:
                    data[:fade_in_samples] *= fade_envelope[:, np.newaxis]
                else:
                    data[:fade_in_samples] *= fade_envelope

                self._in_underrun = False
                self._underrun_fade_pos = 0

            outdata[:] = data
            self._last_good_block = data.copy()

        except queue.Empty:
            self._underrun_count += 1
            self._in_underrun = True

            # Graceful fade-out during underrun
            if self._last_good_block is not None and self._underrun_fade_pos < 8:
                # Exponential fade-out over 8 blocks for smoother decay
                fade_factor = 0.5 ** (self._underrun_fade_pos + 1)
                outdata[:] = self._last_good_block * fade_factor
                self._underrun_fade_pos += 1
            else:
                # Silent output after fade completes
                outdata[:] = 0

        # Final output gain (attenuation only; applied after the DSP limiter so
        # it can never re-introduce clipping). _last_good_block stays pre-gain
        # so volume changes don't compound across underrun recovery.
        if self.volume != 1.0:
            outdata *= self.volume

    def write(self, audio: np.ndarray) -> None:
        """Write audio data to the output buffer.

        Uses a leftover buffer to handle partial blocks efficiently.
        Complete blocks are queued immediately for playback.

        Args:
            audio: Audio data, shape (samples,) or (samples, channels).
        """
        if not self._running:
            return

        # Prepend any leftover samples from previous write
        if self._leftover is not None and len(self._leftover) > 0:
            audio = np.concatenate([self._leftover, audio], axis=0)
            self._leftover = None

        # Queue complete blocks only
        n_complete_blocks = len(audio) // self.blocksize
        for i in range(n_complete_blocks):
            block = audio[i * self.blocksize : (i + 1) * self.blocksize]

            # Ensure correct shape and dtype
            if block.ndim == 1:
                block = np.column_stack([block, block])
            block = block.astype(np.float32)

            self._blocks_written += 1

            # Non-blocking put - drop oldest if full (prevent producer blocking)
            if self._queue.full():
                try:
                    self._queue.get_nowait()  # Drop oldest block
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(block)
            except queue.Full:
                pass  # Should not happen after the above, but safety first

        # Store leftover samples for next write
        leftover_start = n_complete_blocks * self.blocksize
        if leftover_start < len(audio):
            self._leftover = audio[leftover_start:].copy()

        # Start the stream once we have enough buffer pre-filled
        self._maybe_start_stream()

    def flush(self) -> None:
        """Flush any leftover samples with fade-out.

        Call this before stop() to gracefully end playback without clicks.
        """
        if self._leftover is not None and len(self._leftover) > 0:
            # Pad to full block with fade-out
            padded = np.zeros((self.blocksize, self.channels), dtype=np.float32)
            samples_to_copy = min(len(self._leftover), self.blocksize)
            padded[:samples_to_copy] = self._leftover[:samples_to_copy]

            # Apply fade-out to the portion with audio
            if samples_to_copy > 0:
                fade_len = min(256, samples_to_copy)
                fade = np.linspace(1.0, 0.0, fade_len)
                padded[samples_to_copy - fade_len : samples_to_copy] *= fade[:, np.newaxis]

            try:
                self._queue.put_nowait(padded)
            except queue.Full:
                pass
            self._leftover = None

    def stop(self) -> None:
        """Stop the audio output stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._started = False
        self._stream_active = False
        self._leftover = None
        self._last_good_block = None
        self._underrun_fade_pos = 0
        self._recovery_fade_pos = 0
        self._in_underrun = False

    def set_volume(self, volume: float) -> None:
        """Set the output gain.

        Args:
            volume: Gain in [0.0, 1.0]. Attenuation only; values are clamped.
        """
        self.volume = max(0.0, min(1.0, volume))

    def _drain_queue(self) -> None:
        """Discard all buffered blocks (used when pausing)."""
        if self._queue is None:
            return
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def pause(self) -> None:
        """Pause playback by tearing down the stream and dropping the buffer.

        The object stays alive; the next ``write()`` calls re-fill the buffer and
        ``_maybe_start_stream`` recreates the stream once enough is buffered (so
        resume is click-free, just like initial start). Call ``resume()`` to clear
        the pause flag; subsequent ``write()`` calls restart the stream lazily.
        """
        self._paused = True
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._stream_active = False
        self._drain_queue()
        self._leftover = None
        self._in_underrun = False
        self._underrun_fade_pos = 0
        self._recovery_fade_pos = 0

    def resume(self) -> None:
        """Resume playback. The stream is recreated lazily once the buffer re-fills."""
        self._paused = False

    @property
    def underrun_count(self) -> int:
        """Number of buffer underruns detected during playback."""
        return self._underrun_count

    @property
    def queue_level(self) -> int:
        """Current number of blocks in the queue."""
        return self._queue.qsize() if self._queue else 0

    @property
    def buffer_seconds(self) -> float:
        """Current buffer level in seconds."""
        return self.queue_level * self.blocksize / self.sample_rate

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class MockAudioOutput:
    """Mock audio output for testing without sound hardware."""

    def __init__(self, sample_rate: int = 48000, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.written_samples = 0
        self.volume = 1.0
        self._running = False
        self._paused = False
        self._underrun_count = 0

    def start(self) -> None:
        self._running = True
        self._paused = False
        self.written_samples = 0

    def write(self, audio: np.ndarray) -> None:
        if self._running and not self._paused:
            self.written_samples += len(audio)

    def flush(self) -> None:
        pass

    def stop(self) -> None:
        self._running = False

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def underrun_count(self) -> int:
        return self._underrun_count

    @property
    def queue_level(self) -> int:
        return 0

    @property
    def buffer_seconds(self) -> float:
        return 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


@dataclass
class FileAudioOutput:
    """Audio output to a WAV file.

    Collects audio chunks and writes them to a WAV file when stopped.
    """

    filepath: str
    sample_rate: int = 48000
    channels: int = 2

    _buffer: list = field(default_factory=list, init=False, repr=False)
    _running: bool = field(default=False, init=False)

    def start(self) -> None:
        """Start collecting audio."""
        self._running = True
        self._buffer = []

    def write(self, audio: np.ndarray) -> None:
        """Add audio data to the buffer.

        Args:
            audio: Audio data, shape (samples,) or (samples, channels).
        """
        if self._running:
            self._buffer.append(audio.copy())

    def flush(self) -> None:
        """No-op for file output (all data is buffered anyway)."""
        pass

    def stop(self) -> None:
        """Stop collecting and write to WAV file."""
        self._running = False
        if not self._buffer:
            return

        try:
            from scipy.io import wavfile
        except ImportError:
            raise ImportError("scipy is required for file output. Install with: pip install scipy")

        # Concatenate all chunks
        full_audio = np.concatenate(self._buffer, axis=0)

        # Convert to int16 for WAV
        audio_int16 = (full_audio * 32767).astype(np.int16)

        # Write to file
        wavfile.write(self.filepath, self.sample_rate, audio_int16)

        # Clear buffer
        self._buffer = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
