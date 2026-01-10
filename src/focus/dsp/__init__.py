"""DSP module for neural entrainment and audio processing."""

from focus.dsp.dynamics import (
    LimiterState,
    apply_limiter,
    get_true_peak_db,
)
from focus.dsp.entrainment import (
    ModulationState,
    apply_entrainment,
    apply_fade_in,
    apply_fade_out,
    create_test_tone,
)
from focus.dsp.spatial import (
    ReverbState,
    apply_reverb,
    apply_stereo_widening,
)

__all__ = [
    # Entrainment
    "ModulationState",
    "apply_entrainment",
    "apply_fade_in",
    "apply_fade_out",
    "create_test_tone",
    # Spatial
    "ReverbState",
    "apply_reverb",
    "apply_stereo_widening",
    # Dynamics
    "LimiterState",
    "apply_limiter",
    "get_true_peak_db",
]
