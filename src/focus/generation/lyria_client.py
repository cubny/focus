"""Google Lyria RealTime client for music generation.

This module provides a WebSocket client for the Lyria RealTime API,
enabling real-time streaming of AI-generated instrumental music.

Implements overlapping session rotation to avoid the 10-minute API limit
while keeping track-to-track transitions seamless (no audible gap).
"""

import asyncio
import contextlib
import os
import time
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np

# Session rotation constants
# Lyria has a 10-minute session limit; rotate proactively before that.
SESSION_MAX_DURATION_SECONDS = 9 * 60  # 9 minutes

# Seamless rotation: the next session is opened OVERLAP_SECONDS before the
# current one hits its rotation point, warmed up, then the two live streams are
# equal-power crossfaded over CROSSFADE_DURATION_SECONDS. Because the 10-minute
# cap is per-WebSocket, two concurrent sessions overlap without any audio gap.
OVERLAP_SECONDS = 8.0
CROSSFADE_DURATION_SECONDS = 4.0
WARMUP_MIN_SECONDS = 1.5  # Buffered new-session audio before crossfading


# Fraction of the crossfade window over which the OUTGOING track fades to
# silence. Keeping this below 1.0 makes the last part of the crossfade the
# incoming track alone, which avoids a faint "ghost" of the old track (and two
# basslines stacking/beating) right before the switch completes.
CROSSFADE_OUT_END_FRAC = 0.7


def _crossfade_gains(
    start: int, length: int, total: int, out_end_frac: float = CROSSFADE_OUT_END_FRAC
) -> tuple[np.ndarray, np.ndarray]:
    """Crossfade gains for positions ``[start, start+length)`` of ``total``.

    The incoming track rises with an equal-power sine across the full window,
    while the outgoing track falls with a cosine that reaches silence by
    ``out_end_frac`` of the window (and stays there). With ``out_end_frac == 1.0``
    this is a symmetric equal-power crossfade (``fade_out**2 + fade_in**2 == 1``);
    smaller values pull the outgoing track out earlier for a cleaner handover.

    Returns ``(fade_out, fade_in)``.
    """
    idx = np.arange(start, start + length, dtype=np.float64)
    frac = np.clip(idx / max(total, 1), 0.0, 1.0)
    fade_in = np.sin(frac * (0.5 * np.pi))
    out_frac = np.clip(frac / max(out_end_frac, 1e-6), 0.0, 1.0)
    fade_out = np.cos(out_frac * (0.5 * np.pi))
    return fade_out, fade_in


class _ChunkBuffer:
    """A small FIFO of audio chunks with sample-accurate ``take``."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self.total: int = 0

    def add(self, chunk: np.ndarray | None) -> None:
        if chunk is not None and len(chunk) > 0:
            self._chunks.append(chunk)
            self.total += len(chunk)

    def take(self, n: int) -> np.ndarray:
        """Pop and return the first ``n`` samples (concatenated) from the buffer."""
        n = min(n, self.total)
        out: list[np.ndarray] = []
        got = 0
        while got < n and self._chunks:
            head = self._chunks[0]
            remaining = n - got
            if len(head) <= remaining:
                out.append(head)
                got += len(head)
                self._chunks.pop(0)
            else:
                out.append(head[:remaining])
                self._chunks[0] = head[remaining:]
                got += remaining
        self.total -= got
        if not out:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(out, axis=0)


try:
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


@dataclass
class LyriaConfig:
    """Configuration for Lyria music generation."""

    prompt: str
    bpm: int = 120
    temperature: float = 1.0
    guidance: float = 4.0  # 0.0-6.0, higher = stricter adherence to prompt
    density: float = 0.5  # 0.0-1.0, musical density
    brightness: float = 0.5  # 0.0-1.0, tonal quality
    # Lyria outputs 48kHz stereo 16-bit PCM
    sample_rate: int = 48000
    channels: int = 2


class _LiveSession:
    """A single live Lyria WebSocket session with a background reader.

    The session is opened manually (not via ``async with``) so that two
    sessions can be held open simultaneously during an overlapping crossfade.
    A reader task pumps decoded audio chunks into an ``asyncio.Queue`` so the
    consumer can pull from several sessions concurrently without blocking.
    """

    def __init__(self, client: object, config: LyriaConfig, verbose: bool = False) -> None:
        self._client = client
        self._config = config
        self._verbose = verbose
        self._cm = None
        self._session = None
        self._queue: asyncio.Queue | None = None
        self._reader_task: asyncio.Task | None = None
        self.start_time: float = 0.0
        self.error: Exception | None = None
        self._closed = False

    async def open(self) -> None:
        """Connect, configure, and start playback; begin buffering audio."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Realtime music generation is experimental")
            self._cm = self._client.aio.live.music.connect(model="models/lyria-realtime-exp")
            self._session = await self._cm.__aenter__()

        await self._session.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(
                bpm=self._config.bpm,
                temperature=self._config.temperature,
                guidance=self._config.guidance,
                density=self._config.density,
                brightness=self._config.brightness,
            )
        )
        await self._session.set_weighted_prompts(
            prompts=[types.WeightedPrompt(text=self._config.prompt, weight=1.0)]
        )
        await self._session.play()

        self.start_time = time.monotonic()
        self._queue = asyncio.Queue(maxsize=64)
        self._reader_task = asyncio.create_task(self._read_loop())

    def _parse(self, message) -> np.ndarray | None:
        """Extract a normalized float32 audio chunk from a server message."""
        if (
            hasattr(message, "server_content")
            and message.server_content
            and hasattr(message.server_content, "audio_chunks")
            and message.server_content.audio_chunks
        ):
            audio_data = message.server_content.audio_chunks[0].data
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            if self._config.channels == 2 and len(audio_float) >= 2:
                audio_float = audio_float.reshape(-1, 2)
            return audio_float
        return None

    async def _read_loop(self) -> None:
        """Pump decoded audio into the queue until the session ends or errors."""
        try:
            async for message in self._session.receive():
                audio = self._parse(message)
                if audio is not None:
                    await self._queue.put(audio)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surfaced to the consumer via .error
            self.error = e
        finally:
            # Sentinel so consumers waiting on get() always wake up.
            with contextlib.suppress(Exception):
                self._queue.put_nowait(None)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def drain_available(self, buffer: _ChunkBuffer) -> bool:
        """Move all immediately-available chunks into ``buffer`` (non-blocking).

        Returns True if the end-of-stream sentinel was seen.
        """
        ended = False
        if self._queue is None:
            return True
        while True:
            try:
                chunk = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if chunk is None:
                ended = True
                break
            buffer.add(chunk)
        return ended

    async def get(self) -> np.ndarray | None:
        """Await the next audio chunk, or None when the session has ended."""
        if self._queue is None:
            return None
        return await self._queue.get()

    async def set_prompt(self, prompt: str) -> None:
        if self._session:
            try:
                await self._session.set_weighted_prompts(
                    prompts=[types.WeightedPrompt(text=prompt, weight=1.0)]
                )
            except Exception:
                pass  # best-effort

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
        if self._session is not None:
            try:
                await self._session.stop()
            except Exception:
                pass
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None
        self._session = None


@dataclass
class LyriaClient:
    """Client for Google Lyria RealTime API.

    Uses WebSocket connections for real-time streaming music generation.
    Audio is streamed as 16-bit PCM at 48kHz.

    Rotates sessions before the 10-minute API limit using an overlapping
    dual-session crossfade so track transitions are seamless (no audio gap).
    """

    config: LyriaConfig
    session_duration: int = SESSION_MAX_DURATION_SECONDS  # Configurable rotation time
    _client: object = field(default=None, init=False, repr=False)
    _current: _LiveSession | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _session_count: int = field(default=0, init=False, repr=False)
    verbose: bool = field(default=False, init=True)

    def __post_init__(self):
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            )

    async def connect(self, api_key: str | None = None) -> None:
        """Initialize connection to Lyria RealTime API.

        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
        """
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "API key required. Set GOOGLE_API_KEY environment variable or pass api_key."
            )

        # Initialize client with v1alpha API version required for Lyria
        self._client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        self._running = True

    async def _open_with_retry(self) -> _LiveSession | None:
        """Open a new live session, retrying transient errors.

        Returns the session, or None if the caller should fall back to synth
        (model unreachable / retries exhausted / non-retryable error).
        """
        max_retries = 3
        base_delay = 1.0
        retry_count = 0

        while self._running and retry_count <= max_retries:
            try:
                self._session_count += 1
                if self.verbose:
                    if retry_count > 0:
                        print(f"   [Lyria] Retry attempt {retry_count}/{max_retries}...")
                    elif self._session_count > 1:
                        print(f"   [Lyria] Starting session #{self._session_count} (rotation)...")
                    else:
                        print("   [Lyria] Connecting to live session...")

                session = _LiveSession(self._client, self.config, verbose=self.verbose)
                await session.open()
                if self.verbose:
                    print("   [Lyria] Session connected, receiving audio...")
                return session

            except Exception as e:
                error_msg = str(e)

                # Non-retryable: model not found -> fall back to synth
                if "404" in error_msg or "not found" in error_msg.lower():
                    if self.verbose:
                        print(f"   ⚠️  Lyria model unreachable ({error_msg})")
                        print("   🔄 Falling back to Enhanced Synth engine...")
                    return None

                is_retryable = any(
                    indicator in error_msg.lower()
                    for indicator in [
                        "1011",
                        "service",
                        "connection",
                        "websocket",
                        "timeout",
                        "closed",
                    ]
                )
                if is_retryable and retry_count < max_retries:
                    retry_count += 1
                    delay = base_delay * (2 ** (retry_count - 1))
                    if self.verbose:
                        shown = error_msg if len(error_msg) <= 80 else f"{error_msg[:80]}..."
                        print(f"   ⚠️  Lyria connection error: {shown}")
                        print(f"   🔄 Retrying in {delay:.1f}s ({retry_count}/{max_retries})...")
                    await asyncio.sleep(delay)
                    continue

                if self.verbose:
                    if retry_count >= max_retries:
                        print(f"   ❌ Lyria retries exhausted after {max_retries} attempts")
                    else:
                        print(f"   ❌ Lyria error (non-retryable): {error_msg}")
                    print("   🔄 Falling back to Enhanced Synth engine...")
                return None

        return None

    async def generate_stream(self) -> AsyncIterator[np.ndarray]:
        """Generate a continuous music stream with seamless session rotation.

        Yields:
            Audio chunks as numpy arrays, shape (samples, 2) for stereo,
            dtype float32, normalized to [-1, 1].

        Note:
            The next session is opened ahead of time and equal-power crossfaded
            with the current one, so rotations produce no audible gap. Falls
            back to :class:`EnhancedSynthClient` if Lyria is unreachable.
        """
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")

        sample_rate = self.config.sample_rate
        rotate_at = self.session_duration - min(
            OVERLAP_SECONDS, max(1.0, self.session_duration * 0.15)
        )

        use_fallback = False
        current = await self._open_with_retry()
        if current is None:
            use_fallback = True

        try:
            while self._running and not use_fallback:
                self._current = current

                # --- Steady playback until the rotation point ---
                rotate = False
                while self._running:
                    chunk = await current.get()
                    if chunk is None:  # session ended (naturally or via error)
                        break
                    if current.elapsed >= rotate_at:
                        rotate = True
                        break
                    yield chunk

                if not self._running:
                    break

                if not rotate:
                    # Session ended unexpectedly; reconnect (no crossfade
                    # possible since the old stream is already gone).
                    await current.close()
                    current = await self._open_with_retry()
                    if current is None:
                        use_fallback = True
                        break
                    continue

                # --- Open the next session and crossfade into it ---
                new = await self._open_with_retry()
                if new is None:
                    # Could not start the next session; keep the current one
                    # playing to the end, then fall back.
                    async for chunk in self._drain_to_end(current):
                        if not self._running:
                            break
                        yield chunk
                    await current.close()
                    use_fallback = True
                    break

                # Route prompt changes to the incoming session from now on.
                self._current = new

                async for blended in self._crossfade(current, new, sample_rate):
                    if not self._running:
                        break
                    yield blended

                await current.close()
                current = new
        finally:
            if current is not None:
                await current.close()
            self._current = None

        # Fallback to EnhancedSynthClient
        if self._running and use_fallback:
            synth = EnhancedSynthClient(self.config, verbose=self.verbose)
            await synth.connect()
            async for chunk in synth.generate_stream():
                if not self._running:
                    break
                yield chunk

    async def _drain_to_end(self, session: _LiveSession) -> AsyncIterator[np.ndarray]:
        """Yield whatever the session still produces until it ends."""
        while self._running:
            chunk = await session.get()
            if chunk is None:
                return
            yield chunk

    async def _crossfade(
        self, old: _LiveSession, new: _LiveSession, sample_rate: int
    ) -> AsyncIterator[np.ndarray]:
        """Warm up ``new`` while ``old`` keeps playing, then equal-power crossfade.

        Yields the (old audio, then blended, then new audio) so playback stays
        continuous across the whole handover.
        """
        xfade_n = int(CROSSFADE_DURATION_SECONDS * sample_rate)
        warmup_n = int(WARMUP_MIN_SECONDS * sample_rate)

        new_buf = _ChunkBuffer()
        new_ended = False

        # --- Warmup: keep old playing in real time while new fills its buffer ---
        while self._running and new_buf.total < warmup_n and not new_ended:
            new_ended = new.drain_available(new_buf)
            if new_ended or new_buf.total >= warmup_n:
                break
            old_chunk = await old.get()
            if old_chunk is None:
                break
            yield old_chunk

        # --- Crossfade: blend equal amounts of old and new, sample-accurate ---
        old_buf = _ChunkBuffer()
        pos = 0
        old_ended = False
        while self._running and pos < xfade_n:
            new_ended = new.drain_available(new_buf) or new_ended

            if old_buf.total == 0 and not old_ended:
                old_chunk = await old.get()
                if old_chunk is None:
                    old_ended = True
                else:
                    old_buf.add(old_chunk)
            if new_buf.total == 0 and not new_ended:
                new_chunk = await new.get()
                if new_chunk is None:
                    new_ended = True
                else:
                    new_buf.add(new_chunk)

            if old_buf.total == 0 or new_buf.total == 0:
                # One side is exhausted; stop blending and let the tail below run.
                break

            avail = min(old_buf.total, new_buf.total, xfade_n - pos)
            old_seg = old_buf.take(avail)
            new_seg = new_buf.take(avail)
            fade_out, fade_in = _crossfade_gains(pos, avail, xfade_n)
            if old_seg.ndim == 2:
                fade_out = fade_out[:, np.newaxis]
                fade_in = fade_in[:, np.newaxis]
            blended = old_seg * fade_out + new_seg * fade_in
            yield blended.astype(np.float32)
            pos += avail

        # --- Emit any already-buffered new audio so we continue seamlessly ---
        new.drain_available(new_buf)
        if new_buf.total > 0:
            yield new_buf.take(new_buf.total)

    async def stop(self) -> None:
        """Stop the generation stream."""
        self._running = False
        current = self._current
        if current is not None:
            try:
                await current.close()
            except Exception:
                pass
        self._current = None

    async def set_prompt(self, new_prompt: str) -> None:
        """Change the music generation prompt during playback.

        Args:
            new_prompt: The new prompt to guide music generation.
        """
        current = self._current
        if current is not None:
            await current.set_prompt(new_prompt)


class EnhancedSynthClient:
    """Enhanced synthesizer for high-quality ambient drones.

    Used when Lyria API is unavailable. Generates rich ambient textures
    using additive synthesis with detuned oscillators and stereo widening.
    """

    def __init__(self, config: LyriaConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self._running = False
        self._phase = 0.0

        # Stereo delay buffer for click-free stereo widening
        delay_samples = int(0.02 * config.sample_rate)  # 20ms delay
        self._delay_buffer = np.zeros(delay_samples, dtype=np.float32)

        # Oscillator banks for rich texture
        # Base frequencies tailored to the profile's BPM/Mood
        self.oscillators = [
            # Freq multiplier, Amplitude, LFO rate, LFO depth, Phase offset
            (1.0, 0.4, 0.1, 0.002, 0.0),  # Fundamental
            (1.5, 0.2, 0.15, 0.003, 2.0),  # Perfect fifth
            (2.0, 0.15, 0.2, 0.004, 4.0),  # Octave
            (1.01, 0.3, 0.12, 0.002, 1.0),  # Detuned fundamental (chorus effect)
            (1.99, 0.15, 0.18, 0.003, 5.0),  # Detuned octave
        ]

    async def connect(self, api_key: str | None = None) -> None:
        """Initialize synth."""
        self._running = True
        # Reset delay buffer on connect
        delay_samples = int(0.02 * self.config.sample_rate)
        self._delay_buffer = np.zeros(delay_samples, dtype=np.float32)
        if self.verbose:
            print("   [Synth] Lyria unavailable, using enhanced ambient synthesizer")

    async def generate_stream(self) -> AsyncIterator[np.ndarray]:
        """Generate continuous ambient stream."""
        chunk_duration = 0.2  # Short chunks for low latency
        sample_rate = self.config.sample_rate
        chunk_samples = int(chunk_duration * sample_rate)

        t_chunk = np.arange(chunk_samples) / sample_rate

        while self._running:
            # Generate mono mix
            mix = np.zeros(chunk_samples, dtype=np.float32)
            current_t = t_chunk + self._phase

            for mult, amp, lfo_rate, lfo_depth, p_offset in self.oscillators:
                # Apply slow frequency modulation (LFO) for organic movement
                lfo = 1.0 + lfo_depth * np.sin(2 * np.pi * lfo_rate * current_t + p_offset)
                freq = 110.0 * mult * lfo

                # Add sine wave
                mix += amp * np.sin(2 * np.pi * freq * current_t)

            # Stereo widening with proper cross-chunk delay (no clicks)
            # Left channel: original mix
            # Right channel: delayed mix using stateful buffer
            left = mix

            delay_len = len(self._delay_buffer)
            # Concatenate delay buffer with current mix, then split
            delayed_signal = np.concatenate([self._delay_buffer, mix])
            right = delayed_signal[:chunk_samples]
            # Store tail for next chunk
            self._delay_buffer = delayed_signal[-delay_len:].copy()

            # Soft clip limiter
            stereo = np.column_stack([left, right])
            stereo = np.tanh(stereo) * 0.8

            self._phase += chunk_samples / sample_rate

            yield stereo.astype(np.float32)

            # Allow other tasks to run, but don't sleep for duration
            # The output stream blocking write provides the backpressure
            await asyncio.sleep(0)

    async def stop(self) -> None:
        self._running = False

    async def set_prompt(self, new_prompt: str) -> None:
        """Change prompt (no-op for synth, included for API compatibility)."""
        pass  # Synth doesn't support dynamic prompt changes


def create_client(
    config: LyriaConfig,
    use_mock: bool = False,
    verbose: bool = False,
    session_duration: int = SESSION_MAX_DURATION_SECONDS,
) -> LyriaClient | EnhancedSynthClient:
    """Create a Lyria client or fallback to Synth.

    Args:
        config: Configuration for music generation.
        use_mock: If True, force usage of Synth client.
        verbose: If True, enable verbose debug output.
        session_duration: Duration in seconds before session rotation (max 540).

    Returns:
        LyriaClient or EnhancedSynthClient.
    """
    if use_mock or not GENAI_AVAILABLE:
        return EnhancedSynthClient(config, verbose=verbose)

    return LyriaClient(config, session_duration=session_duration, verbose=verbose)
