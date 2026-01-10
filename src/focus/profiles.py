"""Focus profiles for different concentration modes.

Each profile combines:
- A Lyria prompt optimized for focus music generation
- Modulation parameters for neural entrainment
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FocusProfile:
    """Configuration for a focus session."""

    name: str
    description: str

    # Lyria prompt for music generation
    prompt: str

    # Neural entrainment parameters
    modulation_freq: float  # Hz (12-20 for Beta waves)
    modulation_depth: float  # 0.0-1.0

    # Optional Lyria parameters
    bpm: int | None = None
    density: float | None = None  # 0.0-1.0
    brightness: float | None = None  # 0.0-1.0

    # Prompt modifiers for timed sessions (natural musical evolution)
    intro_prompt: str | None = None  # Added during first phase for gradual buildup
    outro_prompt: str | None = None  # Added during last phase for natural wind-down


# Predefined focus profiles
PROFILES: dict[str, FocusProfile] = {
    "deep-work": FocusProfile(
        name="deep-work",
        description="Intense focus for complex tasks requiring deep concentration",
        prompt=(
            "Dark electronic ambient, steady pulse, minimal percussion, "
            "deep bass, broad synthesis, no vocals, consistent energy, "
            "subtle evolving textures, hypnotic rhythm"
        ),
        modulation_freq=18.0,  # High Beta for intense focus
        modulation_depth=0.35,
        bpm=120,
        density=0.4,
        brightness=0.3,
        intro_prompt=(
            "emerging from silence, slowly building, sparse beginning, "
            "gradually introducing elements"
        ),
        outro_prompt=(
            "slowly fading, elements dropping away, peaceful resolution, "
            "gentle ending"
        ),
    ),
    "light-study": FocusProfile(
        name="light-study",
        description="Gentle focus for reading, studying, or light work",
        prompt=(
            "Lo-fi beats, warm piano chords, gentle rhythm, "
            "ambient texture, soft vinyl crackle, no vocals, "
            "cozy atmosphere, slow tempo"
        ),
        modulation_freq=12.0,  # Low Beta for relaxed attention
        modulation_depth=0.25,
        bpm=85,
        density=0.3,
        brightness=0.5,
        intro_prompt="soft opening, delicate start, notes appearing gently, warmth growing",
        outro_prompt="winding down, becoming quieter, peaceful fade, restful conclusion",
    ),
    "adhd-support": FocusProfile(
        name="adhd-support",
        description="Structured rhythm to help maintain attention with ADHD",
        prompt=(
            "Ambient techno, consistent driving beat, steady energy, "
            "broad synthesis, layered textures, no vocals, "
            "predictable progression, subtle pink noise undertone"
        ),
        modulation_freq=15.0,  # Mid Beta
        modulation_depth=0.40,  # Slightly stronger modulation
        bpm=128,
        density=0.5,
        brightness=0.4,
        intro_prompt="rhythm emerging, beat building gradually, layers joining one by one",
        outro_prompt="beat softening, layers dissolving, energy releasing, calm ending",
    ),
    "creative-flow": FocusProfile(
        name="creative-flow",
        description="Open-ended focus for creative and exploratory work",
        prompt=(
            "Atmospheric ambient, evolving soundscapes, ethereal pads, "
            "gentle movement, space and reverb, no vocals, "
            "dreamy textures, organic sounds"
        ),
        modulation_freq=10.0,  # Alpha-Beta border for creative state
        modulation_depth=0.20,
        bpm=90,
        density=0.25,
        brightness=0.6,
        intro_prompt="dreamy awakening, textures slowly materializing, ethereal beginning",
        outro_prompt="drifting away, textures dissolving, peaceful floating, serene ending",
    ),
    "energetic": FocusProfile(
        name="energetic",
        description="High-energy focus for tasks requiring momentum and drive",
        prompt=(
            "Driving electronic, punchy drums, energetic bassline, "
            "bright synth leads, powerful rhythms, no vocals, "
            "dynamic tension, forward momentum, bold synthesis"
        ),
        modulation_freq=20.0,  # High Beta for maximum alertness
        modulation_depth=0.40,
        bpm=140,
        density=0.6,
        brightness=0.7,
        intro_prompt=(
            "energy igniting, rhythm sparking to life, power building "
            "from the ground up"
        ),
        outro_prompt=(
            "energy channeling down, rhythm easing back, power settling "
            "into calm strength"
        ),
    ),
    "uplifting": FocusProfile(
        name="uplifting",
        description="Positive, bright focus for maintaining optimism and motivation",
        prompt=(
            "Uplifting melodic ambient, warm major chords, bright arpeggios, "
            "hopeful progressions, shimmering pads, no vocals, "
            "positive energy, gentle euphoria, inspiring textures"
        ),
        modulation_freq=14.0,  # Mid Beta for balanced, positive focus
        modulation_depth=0.30,
        bpm=110,
        density=0.45,
        brightness=0.8,
        intro_prompt="light dawning, warmth spreading gently, hope awakening softly",
        outro_prompt="light softening into golden glow, warmth lingering, peaceful gratitude",
    ),
}


def get_profile(name: str) -> FocusProfile:
    """Get a focus profile by name.

    Args:
        name: Profile name (e.g., 'deep-work', 'light-study').

    Returns:
        The corresponding FocusProfile.

    Raises:
        KeyError: If the profile name is not found.
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise KeyError(f"Unknown profile '{name}'. Available profiles: {available}")
    return PROFILES[name]


def list_profiles() -> list[FocusProfile]:
    """Get all available focus profiles.

    Returns:
        List of all FocusProfile objects.
    """
    return list(PROFILES.values())
