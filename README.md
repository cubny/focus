# Focus Music PoC

A proof-of-concept for generating focus-enhancing music using neural entrainment.

## How It Works

1. **Google Lyria RealTime** generates continuous, non-looping instrumental music
2. **DSP post-processing** applies amplitude modulation (12-20 Hz) for neural entrainment
3. **Focus profiles** combine genre prompts with optimized modulation settings

## Setup

```bash
# Install with pip (Python 3.12+ required)
pip install -e ".[dev]"

# Set your Google API key
export GOOGLE_API_KEY="your-api-key-from-aistudio"
```

## Usage

```bash
# Start a focus session
focus start --profile deep-work

# List available profiles
focus profiles

# Custom settings
focus start --frequency 16 --depth 0.3 --prompt "ambient electronic..."
```

## Profiles

| Profile | Frequency | Description |
|---------|-----------|-------------|
| deep-work | 18 Hz | Dark electronic, intense focus |
| light-study | 12 Hz | Lo-fi beats, light concentration |
| adhd-support | 15 Hz | Consistent energy with pink noise |

## Development

```bash
# Run tests
pytest

# Format code
ruff format src/ tests/
ruff check --fix src/ tests/
```

## Recording & Verification

To verify the neural entrainment modulation is working correctly, you can record the audio output using the `--output` flag and analyze it with the built-in `analyze` command.

### Recording Audio

Use the `--output` / `-o` flag to save your focus session to a WAV file:

```bash
# Record a 30-second session with the deep-work profile
focus start --profile deep-work --duration 30 --output session.wav

# Short form
focus start -p deep-work -o session.wav --duration 30
```

The audio is saved alongside playback - you'll hear the music while it records.

### Analyzing Output

Use `focus analyze` to verify the modulation in your recorded file:

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

