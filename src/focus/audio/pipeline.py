"""Real-time audio processing pipeline.

Connects the Lyria generator to the DSP entrainment layer and audio output.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field  
from typing import Protocol

import numpy as np

from focus.dsp.entrainment import ModulationState, apply_entrainment
from focus.profiles import FocusProfile


class AudioGenerator(Protocol):
    """Protocol for audio generators (Lyria client or mock)."""

    async def connect(self, api_key: str | None = None) -> None: ...
    async def generate_stream(self): ...
    async def stop(self) -> None: ...


@dataclass
class AudioPipeline:
    """Real-time audio processing pipeline.

    Receives audio from a generator, applies neural entrainment,
    and outputs to audio callback.
    """

    generator: AudioGenerator
    profile: FocusProfile
    output_callback: Callable[[np.ndarray], None]
    sample_rate: int = 48000

    _running: bool = field(default=False, init=False)
    _mod_state: ModulationState = field(default_factory=ModulationState, init=False)
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
        """Main processing loop: receive -> modulate -> output."""
        async for chunk in self.generator.generate_stream():
            if not self._running:
                break

            # Apply neural entrainment
            modulated, self._mod_state = apply_entrainment(
                chunk,
                self.sample_rate,
                target_freq=self.profile.modulation_freq,
                depth=self.profile.modulation_depth,
                state=self._mod_state,
            )

            # Send to output
            self.output_callback(modulated)

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
