"""Google Lyria RealTime client for music generation.

This module provides a WebSocket client for the Lyria RealTime API,
enabling real-time streaming of AI-generated instrumental music.

Implements proactive session rotation to avoid the 10-minute API limit.
"""

import asyncio
import os
import time
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np

# Session rotation constants
# Lyria has a 10-minute session limit; rotate proactively at 9 minutes
SESSION_MAX_DURATION_SECONDS = 9 * 60  # 9 minutes
CROSSFADE_DURATION_SECONDS = 3.0  # Seconds to buffer for seamless transition


def _apply_crossfade(
    old_audio: np.ndarray, new_audio: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Apply crossfade between two audio chunks for seamless transition.

    Args:
        old_audio: Tail of the previous session's audio (stereo, float32)
        new_audio: Head of the new session's audio (stereo, float32)
        sample_rate: Audio sample rate (e.g., 48000)

    Returns:
        Crossfaded audio chunk.
    """
    crossfade_samples = int(CROSSFADE_DURATION_SECONDS * sample_rate)

    # Ensure we have enough samples for crossfade
    old_samples = min(len(old_audio), crossfade_samples)
    new_samples = min(len(new_audio), crossfade_samples)
    overlap_samples = min(old_samples, new_samples)

    if overlap_samples == 0:
        return new_audio

    # Create fade curves (linear for simplicity and low CPU)
    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)

    # Apply fades to stereo audio
    if old_audio.ndim == 2:
        fade_out = fade_out[:, np.newaxis]
        fade_in = fade_in[:, np.newaxis]

    # Blend the overlapping region
    old_tail = old_audio[-overlap_samples:] * fade_out
    new_head = new_audio[:overlap_samples] * fade_in
    blended = old_tail + new_head

    # Return: blended overlap + rest of new audio
    if overlap_samples < len(new_audio):
        return np.concatenate([blended, new_audio[overlap_samples:]], axis=0)
    return blended

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


@dataclass
class LyriaClient:
    """Client for Google Lyria RealTime API.

    Uses WebSocket connection for real-time streaming music generation.
    Audio is streamed as 16-bit PCM at 48kHz.

    Implements proactive session rotation to avoid the 10-minute
    API limit which causes audio degradation.
    """

    config: LyriaConfig
    session_duration: int = SESSION_MAX_DURATION_SECONDS  # Configurable rotation time
    _client: object = field(default=None, init=False, repr=False)
    _session: object = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _session_start_time: float = field(default=0.0, init=False, repr=False)
    _crossfade_buffer: np.ndarray | None = field(default=None, init=False, repr=False)
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

    async def generate_stream(self) -> AsyncIterator[np.ndarray]:
        """Generate continuous music stream with automatic session rotation.

        Yields:
            Audio chunks as numpy arrays, shape (samples, 2) for stereo,
            dtype float32, normalized to [-1, 1].

        Note:
            Proactively rotates sessions at 9 minutes to avoid the 10-minute
            API limit which causes audio degradation. Crossfades between
            sessions for seamless transitions.
        """
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")

        max_retries = 3
        use_fallback = False

        # Main loop: rotates sessions when approaching time limit
        while self._running and not use_fallback:
            retry_count = 0
            base_delay = 1.0
            session_needs_rotation = False

            while retry_count <= max_retries and self._running and not session_needs_rotation:
                try:
                    self._session_count += 1
                    if self.verbose:
                        if retry_count > 0:
                            print(f"   [Lyria] Retry attempt {retry_count}/{max_retries}...")
                        elif self._session_count > 1:
                            print(
                                f"   [Lyria] Starting session #{self._session_count} (rotation)..."
                            )
                        else:
                            print("   [Lyria] Connecting to live session...")

                    # Connect to Lyria music model
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message="Realtime music generation is experimental"
                        )
                        async with self._client.aio.live.music.connect(
                            model="models/lyria-realtime-exp",
                        ) as session:
                            self._session = session
                            self._session_start_time = time.monotonic()
                            retry_count = 0  # Reset on successful connection

                            if self.verbose:
                                print("   [Lyria] Session connected")

                            # Configure music generation parameters
                            await session.set_music_generation_config(
                                config=types.LiveMusicGenerationConfig(
                                    bpm=self.config.bpm,
                                    temperature=self.config.temperature,
                                    guidance=self.config.guidance,
                                    density=self.config.density,
                                    brightness=self.config.brightness,
                                )
                            )

                            # Set the music style prompt
                            await session.set_weighted_prompts(
                                prompts=[types.WeightedPrompt(text=self.config.prompt, weight=1.0)]
                            )

                            # Start playback
                            await session.play()
                            if self.verbose:
                                print("   [Lyria] Playback started, receiving audio...")

                            # Receive audio chunks
                            async for message in session.receive():
                                if not self._running:
                                    return  # Clean exit

                                # Check session duration for rotation
                                elapsed = time.monotonic() - self._session_start_time
                                if elapsed >= self.session_duration:
                                    if self.verbose:
                                        print(
                                            f"   [Lyria] Session #{self._session_count} "
                                            f"reached {elapsed:.0f}s, rotating..."
                                        )
                                    session_needs_rotation = True
                                    break  # Exit receive loop to rotate

                                # Check for audio data in the message
                                if (
                                    hasattr(message, "server_content")
                                    and message.server_content
                                    and hasattr(message.server_content, "audio_chunks")
                                    and message.server_content.audio_chunks
                                ):
                                    # Get raw audio data from the chunk
                                    audio_data = message.server_content.audio_chunks[0].data

                                    # Raw 16-bit PCM audio
                                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                                    audio_float = audio_int16.astype(np.float32) / 32768.0

                                    # Reshape to stereo (interleaved L/R samples)
                                    if self.config.channels == 2 and len(audio_float) >= 2:
                                        audio_float = audio_float.reshape(-1, 2)

                                    # Apply crossfade from previous session
                                    if self._crossfade_buffer is not None:
                                        audio_float = _apply_crossfade(
                                            self._crossfade_buffer,
                                            audio_float,
                                            self.config.sample_rate,
                                        )
                                        self._crossfade_buffer = None  # Clear buffer

                                    # Buffer tail ONLY when approaching rotation
                                    # to avoid crossfading continuous playback
                                    time_until_rotation = (
                                        self.session_duration - elapsed
                                    )
                                    if time_until_rotation <= 5.0:  # Last 5 seconds
                                        crossfade_samples = int(
                                            CROSSFADE_DURATION_SECONDS
                                            * self.config.sample_rate
                                        )
                                        if len(audio_float) >= crossfade_samples:
                                            tail = audio_float[-crossfade_samples:]
                                            self._crossfade_buffer = tail.copy()

                                    yield audio_float

                            # If session ended cleanly without rotation, we're done
                            if not session_needs_rotation:
                                return

                except Exception as e:
                    error_msg = str(e)

                    # Check for non-retryable errors (model not found)
                    if "404" in error_msg or "not found" in error_msg.lower():
                        if self.verbose:
                            print(f"   ⚠️  Lyria model unreachable ({error_msg})")
                            print("   🔄 Falling back to Enhanced Synth engine...")
                        use_fallback = True
                        break

                    # Check for retryable errors
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
                            print(
                                f"   ⚠️  Lyria connection error: {error_msg[:80]}..."
                                if len(error_msg) > 80
                                else f"   ⚠️  Lyria connection error: {error_msg}"
                            )
                            print(
                                f"   🔄 Retrying in {delay:.1f}s "
                                f"({retry_count}/{max_retries})..."
                            )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        if self.verbose:
                            if retry_count >= max_retries:
                                print(f"   ❌ Lyria retries exhausted after {max_retries} attempts")
                            else:
                                print(f"   ❌ Lyria error (non-retryable): {error_msg}")
                            print("   🔄 Falling back to Enhanced Synth engine...")
                        use_fallback = True
                        break

            # Yield transition audio during session rotation gap
            # This fills the silence while the new session connects
            if session_needs_rotation and self._crossfade_buffer is not None:
                if self.verbose:
                    print("   [Lyria] Playing transition audio...")
                # Apply fade-out to transition buffer and yield it
                # This provides audio continuity during connection
                fade_samples = len(self._crossfade_buffer)
                fade_out = np.linspace(1.0, 0.3, fade_samples, dtype=np.float32)
                if self._crossfade_buffer.ndim == 2:
                    fade_out = fade_out[:, np.newaxis]
                transition_audio = self._crossfade_buffer * fade_out
                yield transition_audio.astype(np.float32)
                # Keep buffer for crossfade with new session's first chunk
                # (but attenuated to avoid double-loud join)
                self._crossfade_buffer = (self._crossfade_buffer * 0.3).astype(np.float32)

        # Fallback to EnhancedSynthClient
        if self._running and use_fallback:
            synth = EnhancedSynthClient(self.config, verbose=self.verbose)
            await synth.connect()
            async for chunk in synth.generate_stream():
                if not self._running:
                    break
                yield chunk

    async def stop(self) -> None:
        """Stop the generation stream."""
        self._running = False
        if self._session:
            try:
                await self._session.stop()
            except Exception:
                pass
            self._session = None

    async def set_prompt(self, new_prompt: str) -> None:
        """Change the music generation prompt during playback.

        Args:
            new_prompt: The new prompt to guide music generation.
        """
        if self._session:
            try:
                await self._session.set_weighted_prompts(
                    prompts=[types.WeightedPrompt(text=new_prompt, weight=1.0)]
                )
            except Exception:
                pass  # Ignore errors, prompt change is best-effort


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
