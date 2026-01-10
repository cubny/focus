"""Audio output using sounddevice for real-time playback."""

import queue
from dataclasses import dataclass, field

import numpy as np

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


@dataclass
class AudioOutput:
    """Real-time audio output using sounddevice.

    Uses a continuous sample buffer to avoid zero-padding clicks.
    Leftover samples are held until enough data arrives for a complete block.
    """

    sample_rate: int = 48000
    channels: int = 2
    blocksize: int = 2048
    buffersize: int = 50  # Queue capacity in blocks
    prefill_blocks: int = 10  # Pre-fill this many blocks before starting actual playback

    _stream: object = field(default=None, init=False, repr=False)
    _queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=50), init=False, repr=False
    )
    _running: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)
    _leftover: np.ndarray | None = field(default=None, init=False, repr=False)
    _underrun_count: int = field(default=0, init=False)
    _last_block: np.ndarray | None = field(default=None, init=False, repr=False)
    _fade_position: int = field(default=0, init=False)
    _blocks_written: int = field(default=0, init=False)

    def __post_init__(self):
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice package is required. Install with: pip install sounddevice"
            )
        self._queue = queue.Queue(maxsize=self.buffersize)

    def start(self) -> None:
        """Start the audio output stream."""
        self._running = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._started = True

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Callback for sounddevice to get audio data."""
        if status and status.output_underflow:
            self._underrun_count += 1

        try:
            data = self._queue.get_nowait()
            outdata[:] = data
            self._last_block = data.copy()
            self._fade_position = 0

        except queue.Empty:
            self._underrun_count += 1

            # Graceful underrun: fade to silence using last block
            if self._last_block is not None and self._fade_position < 4:
                fade_factor = max(0.0, 1.0 - (self._fade_position + 1) * 0.25)
                outdata[:] = self._last_block * fade_factor
                self._fade_position += 1
            else:
                outdata[:] = 0

    def write(self, audio: np.ndarray) -> None:
        """Write audio data to the output buffer.

        Uses a leftover buffer to avoid zero-padding. Only complete blocks
        are queued; incomplete data is held for the next write() call.

        Args:
            audio: Audio data, shape (samples, channels).
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

            # Use put with timeout to avoid deadlock but maintain backpressure
            try:
                self._queue.put(block, timeout=0.1)
            except queue.Full:
                # If queue is full, skip this block (better than blocking forever)
                pass

        # Store leftover samples for next write
        leftover_start = n_complete_blocks * self.blocksize
        if leftover_start < len(audio):
            self._leftover = audio[leftover_start:].copy()

    def flush(self) -> None:
        """Flush any leftover samples with fade-out.

        Call this before stop() to gracefully end playback without clicks.
        """
        if self._leftover is not None and len(self._leftover) > 0:
            # Pad to full block with fade-out
            padded = np.zeros((self.blocksize, self.channels), dtype=np.float32)
            samples_to_copy = min(len(self._leftover), self.blocksize)
            padded[:samples_to_copy] = self._leftover[:samples_to_copy]

            # Apply fade-out to the padded portion
            fade_len = self.blocksize - samples_to_copy
            if fade_len > 0 and samples_to_copy > 0:
                fade = np.linspace(1.0, 0.0, min(256, samples_to_copy))
                if len(fade) <= samples_to_copy:
                    padded[samples_to_copy-len(fade):samples_to_copy] *= fade[:, np.newaxis]

            try:
                self._queue.put(padded, timeout=0.1)
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
        self._leftover = None
        self._last_block = None
        self._fade_position = 0

    @property
    def underrun_count(self) -> int:
        """Number of buffer underruns detected during playback."""
        return self._underrun_count

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
        self._running = False

    def start(self) -> None:
        self._running = True
        self.written_samples = 0

    def write(self, audio: np.ndarray) -> None:
        if self._running:
            self.written_samples += len(audio)

    def stop(self) -> None:
        self._running = False

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

    def stop(self) -> None:
        """Stop collecting and write to WAV file."""
        self._running = False
        if not self._buffer:
            return

        try:
            from scipy.io import wavfile
        except ImportError:
            raise ImportError(
                "scipy is required for file output. Install with: pip install scipy"
            )

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
