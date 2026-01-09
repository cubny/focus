"""Real-time audio processing pipeline.

Connects the Lyria generator to the DSP entrainment layer and audio output.
Includes spatial effects, dynamics processing, and glitch-free chunk handling.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field  
from typing import Protocol

import numpy as np

from focus.dsp.entrainment import ModulationState, apply_entrainment
from focus.dsp.spatial import ReverbState, apply_reverb, apply_stereo_widening
from focus.dsp.dynamics import LimiterState, apply_limiter
from focus.profiles import FocusProfile


class AudioGenerator(Protocol):
    """Protocol for audio generators (Lyria client or mock)."""

    async def connect(self, api_key: str | None = None) -> None: ...
    async def generate_stream(self): ...
    async def stop(self) -> None: ...


@dataclass
class OverlapAddState:
    """State for overlap-add processing to eliminate chunk boundary glitches."""
    
    overlap_samples: int = 256
    _prev_tail: np.ndarray | None = field(default=None, init=False, repr=False)
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply overlap-add crossfade at chunk boundaries.
        
        Args:
            audio: Input audio chunk, shape (samples,) or (samples, channels).
        
        Returns:
            Processed audio with smooth transitions.
        """
        if self._prev_tail is None:
            # First chunk - just store tail and return
            self._prev_tail = audio[-self.overlap_samples:].copy()
            return audio
        
        n_overlap = min(self.overlap_samples, len(audio), len(self._prev_tail))
        
        if n_overlap == 0:
            self._prev_tail = audio[-self.overlap_samples:].copy()
            return audio
        
        # Create raised-cosine crossfade envelope
        t = np.linspace(0, np.pi / 2, n_overlap)
        fade_in = np.sin(t) ** 2
        fade_out = np.cos(t) ** 2
        
        # Reshape for stereo
        if audio.ndim == 2:
            fade_in = fade_in[:, np.newaxis]
            fade_out = fade_out[:, np.newaxis]
        
        # Apply crossfade to beginning of current chunk
        output = audio.copy()
        output[:n_overlap] = (
            self._prev_tail[-n_overlap:] * fade_out + 
            audio[:n_overlap] * fade_in
        )
        
        # Store tail for next chunk
        self._prev_tail = audio[-self.overlap_samples:].copy()
        
        return output
    
    def reset(self):
        """Clear the overlap buffer."""
        self._prev_tail = None


@dataclass
class AudioPipeline:
    """Real-time audio processing pipeline.

    Receives audio from a generator, applies neural entrainment,
    spatial effects, dynamics limiting, and outputs to audio callback.
    
    DSP chain order:
    1. Neural entrainment (amplitude modulation)
    2. Spatialization (reverb + stereo widening)
    3. Overlap-add crossfade (glitch removal)
    4. Dynamics limiting (True Peak safety)
    """

    generator: AudioGenerator
    profile: FocusProfile
    output_callback: Callable[[np.ndarray], None]
    sample_rate: int = 48000
    
    # Effect enable flags
    enable_reverb: bool = True
    enable_stereo_widening: bool = True
    enable_limiter: bool = True
    stereo_width: float = 1.2  # 1.0 = original, 1.2 = slightly wider

    _running: bool = field(default=False, init=False)
    _mod_state: ModulationState = field(default_factory=ModulationState, init=False)
    _reverb_state: ReverbState | None = field(default=None, init=False)
    _limiter_state: LimiterState | None = field(default=None, init=False)
    _overlap_state: OverlapAddState = field(default_factory=OverlapAddState, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)

    async def start(self, api_key: str | None = None) -> None:
        """Start the audio pipeline.

        Args:
            api_key: Google API key for Lyria. If not provided, uses env var.
        """
        await self.generator.connect(api_key)
        self._running = True
        self._task = asyncio.create_task(self._process_loop())

    async def _process_loop(self) -> None:
        """Main processing loop: receive -> modulate -> spatial -> limit -> output."""
        async for chunk in self.generator.generate_stream():
            if not self._running:
                break

            processed = chunk
            
            # 1. Apply neural entrainment
            processed, self._mod_state = apply_entrainment(
                processed,
                self.sample_rate,
                target_freq=self.profile.modulation_freq,
                depth=self.profile.modulation_depth,
                state=self._mod_state,
            )

            # 2. Apply spatialization
            if self.enable_reverb:
                processed, self._reverb_state = apply_reverb(
                    processed,
                    self.sample_rate,
                    room_size=0.3,
                    damping=0.5,
                    wet_dry_mix=0.15,
                    state=self._reverb_state,
                )
            
            if self.enable_stereo_widening and processed.ndim == 2:
                processed = apply_stereo_widening(processed, width=self.stereo_width)
            
            # 3. Apply overlap-add for glitch-free transitions
            processed = self._overlap_state.process(processed)
            
            # 4. Apply dynamics limiting (last in chain)
            if self.enable_limiter:
                processed, self._limiter_state = apply_limiter(
                    processed,
                    self.sample_rate,
                    ceiling_db=-0.1,
                    state=self._limiter_state,
                )

            # Send to output
            self.output_callback(processed)

    async def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._running = False
        await self.generator.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def update_profile(self, profile: FocusProfile) -> None:
        """Update the focus profile mid-session.

        This changes the modulation parameters immediately.
        Note: Lyria prompt changes require reconnection.
        """
        self.profile = profile

    @property
    def is_running(self) -> bool:
        """Check if the pipeline is active."""
        return self._running
