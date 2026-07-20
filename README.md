# Focus Music

A proof-of-concept for generating focus-enhancing music using neural entrainment.
![](https://github.com/cubny/focus/blob/main/rec.gif)

## How It Works

1. **Google Lyria RealTime** generates continuous, non-looping instrumental music
2. **DSP post-processing** applies amplitude modulation (12-20 Hz) for neural entrainment
3. **Spatialization** adds subtle reverb and stereo widening for a "premium" feel
4. **Focus profiles** combine genre prompts with optimized modulation settings

## Audio Enhancements

The system includes professional audio processing to enhance the listening experience:

- **Spatial Ambience**: Schroeder reverb algorithm creates a comfortable acoustic space
- **Stereo Widening**: M/S processing for a wider, more immersive soundstage
- **Safety Limiter**: True Peak limiting prevents digital clipping and ensures consistent volume

## Installation

### Quick Install (Recommended)

Run the installer script to automatically set up Focus Music and its dependencies:

```bash
curl -LsSf https://raw.githubusercontent.com/cubny/focus/main/install.sh | bash
```

This will:
- Install [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- Install PortAudio (audio library) if needed
- Install Focus Music globally

After installation, set your API key and start a session:

```bash
export GOOGLE_API_KEY="your-api-key"  # Get one at https://aistudio.google.com/
focus start --profile deep-work
```

### Manual Installation with uv

If you prefer to install manually:

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install focus as a CLI tool
uv tool install focus-music

# 3. Set your API key and run
export GOOGLE_API_KEY="your-api-key"
focus start --profile deep-work
```

### One-liner (Try without installing)

```bash
# Run focus directly without permanent installation
uvx focus-music start --profile deep-work
```

<details>
<summary>🛠️ Developer Setup</summary>

For development, clone the repository and use `uv sync`:

```bash
git clone https://github.com/cubny/focus.git
cd focus

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Run the CLI
uv run focus start --profile deep-work

# Or activate the virtual environment
source .venv/bin/activate
focus start --profile deep-work
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/
```

</details>

<details>
<summary>📦 Alternative: pip installation</summary>

If you prefer using pip directly:

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install from PyPI (when published)
pip install focus-music

# Or install from source
git clone https://github.com/cubny/focus.git
cd focus
pip install -e ".[dev]"
```

</details>

### Prerequisites

- **macOS** or **Linux** (Windows via WSL)
- **Google Gemini API key** (free from [AI Studio](https://aistudio.google.com/))
- Python 3.12+ (automatically installed by uv if missing)

## Usage

The fastest way to start: just run `focus` in a terminal. With no arguments it
opens an interactive picker — use ↑/↓ (or number keys) to choose a profile and
Enter to start. No need to memorize profile names or flags.

```bash
# Interactive launcher (pick a profile, then play)
focus

# Start a focus session with default profile
focus start --profile deep-work

# List available profiles
focus profiles
```

### Playback controls

While a session is playing in a terminal, single keys control playback (a status
line at the bottom shows what's playing and the available keys):

| Key | Action |
|-----|--------|
| `space` / `p` | Pause / resume (reconnects with a fresh take on resume) |
| `n` | Next take — regenerate fresh music in the same profile |
| `↑` / `↓` (or `+` / `-`) | Volume up / down |
| `?` | Toggle expanded key help |
| `q` | Quit |
| `Ctrl+C` | Hard stop |

These controls activate only when running attached to a terminal; piped or
recording-only runs (`-o file.wav` in a script) behave exactly as before.

```bash
# Custom modulation settings
focus start --frequency 16 --depth 0.3 --prompt "ambient electronic..."

# Timed session (must be at least 60 seconds)
focus start --profile deep-work --duration 1200  # 20 minutes

# Record to file (plays audio while recording)
focus start --profile deep-work --duration 300 -o session.wav
```

### Audio Enhancement Options

```bash
# Adjust stereo width (1.0 = normal, >1.0 = wider)
focus start --stereo-width 1.5

# Disable reverb for a drier sound
focus start --no-reverb

# Disable the safety limiter (for raw, unprocessed dynamics)
focus start --no-limiter

# Minimal processing (dry, no limiting)
focus start --no-reverb --no-limiter

# Maximum enhancement
focus start --reverb --stereo-width 2.0 --limiter

# Debug mode with verbose output
focus start -v
```

## Musical Evolution (Timed Sessions)

When you specify a `--duration` for your session, the music automatically evolves through three distinct phases to support your workflow:

1. **Intro (15s)**: The music starts sparse and gradually builds up in density ("emerging from silence"), helping you eased into focus.
2. **Main**: The core focus session uses the full profile settings for deep work.
3. **Outro (30s)**: The music naturally winds down ("peaceful resolution") before the session ends.

This creates a narrative arc for your work session rather than just cutting off abruptly.

```bash
# Start a 20-minute timed session with full musical evolution
focus start --profile deep-work --duration 1200
```

## Profiles

| Profile | Frequency | Description |
|---------|-----------|-------------|
| deep-work | 18 Hz | Dark electronic, intense focus |
| light-study | 12 Hz | Lo-fi beats, light concentration |
| adhd-support | 15 Hz | Consistent energy with pink noise |
| creative-flow | 10 Hz | Evolving ambient soundscapes (Alpha/Beta border) |
| energetic | 20 Hz | Driving electronic, high-energy momentum |
| uplifting | 14 Hz | Warm melodic ambient, positive motivation |

## Audio Demos

Listen to samples generated by the system (best experienced with headphones):

### Deep Work (18 Hz Beta)
Dark electronic for intense focus.

🎧 [Listen on SoundCloud](https://soundcloud.com/cubny/deepwork-300-reverb-limiter-width-2-1)

### Light Study (12 Hz Alpha)
Lo-fi beats for reading and light tasks.

🎧 [Listen on SoundCloud](https://soundcloud.com/cubny/light-study-300-reverb-limiter-width-2-2)

### Creative Flow (10 Hz Alpha)
Evolving ambient soundscapes.

🎧 [Listen on SoundCloud](https://soundcloud.com/cubny/creative-flow-300-reverb-limiter-width-2-3)

### Energetic (20 Hz Beta)
High-energy momentum.

🎧 [Listen on SoundCloud](https://soundcloud.com/cubny/energetic-1-120-reverb-limiter-width-2-5)

### Uplifting (14 Hz Alpha/Beta)
Warm, positive motivation.

🎧 [Listen on SoundCloud](https://soundcloud.com/cubny/uplifting-1-1200-reverb-limiter-width-2-4)

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/
```

## Verification

Use `focus analyze` to verify the neural entrainment modulation in a recorded session:

```bash
# Analyze with default expected frequency (15 Hz)
focus analyze session.wav

# Analyze with specific expected modulation parameters
focus analyze session.wav --expected-freq 18 --expected-depth 0.3

# Example for deep-work profile (18 Hz modulation)
focus analyze session.wav -f 18 -d 0.3
```

The analyze command will show:
- Detected modulation frequency in the audio
- Peak amplitude at the expected frequency
- Whether the modulation matches the expected parameters
- A frequency spectrum visualization

