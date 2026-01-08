"""Audio output using sounddevice for real-time playback."""

import queue
import threading
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

    Uses a queue-based approach for thread-safe audio buffering.
    """

    sample_rate: int = 48000
    channels: int = 2
    blocksize: int = 2048
    buffersize: int = 50  # Increased buffer: ~2 seconds

    _stream: object = field(default=None, init=False, repr=False)
    _queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=50), init=False, repr=False)
    _running: bool = field(default=False, init=False)

    def __post_init__(self):
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice package is required. Install with: pip install sounddevice"
            )
        # Re-initialize queue with maxsize for blocking put behavior
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

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Callback for sounddevice to get audio data."""
        if status:
            pass  # Could log underruns here

        try:
            data = self._queue.get_nowait()
            # Ensure data fits the output buffer
            if len(data) >= frames:
                outdata[:] = data[:frames]
            else:
                outdata[: len(data)] = data
                outdata[len(data) :] = 0
        except queue.Empty:
            outdata[:] = 0  # Output silence if buffer is empty

    def write(self, audio: np.ndarray) -> None:
        """Write audio data to the output buffer.

        Args:
            audio: Audio data, shape (samples, channels).
        """
        if not self._running:
            return

        # Split into blocks and queue
        for i in range(0, len(audio), self.blocksize):
            block = audio[i : i + self.blocksize]
            if block.shape[0] < self.blocksize:
                # Pad the last block
                padded = np.zeros((self.blocksize, self.channels), dtype=audio.dtype)
                padded[: block.shape[0]] = block
                block = padded

            # Put in queue, blocking if full (backpressure)
            self._queue.put(block)

    def stop(self) -> None:
        """Stop the audio output stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

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
