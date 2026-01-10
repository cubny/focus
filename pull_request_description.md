# PR: Phase 3 Audio Quality & Spatial Enhancement 🔊

## Description
This PR implements **Phase 3** of the roadmap, focusing on professional audio processing to elevate the listening experience and reduce mental fatigue during long focus sessions. It introduces spatial effects, dynamics processing, and glitch-free chunk transitions.

## Key Changes

### 1. Spatialization (`src/focus/dsp/spatial.py`)
- **Schroeder Reverb**: Implemented a lightweight parallel-comb/series-allpass reverberator for subtle room ambience.
- **Stereo Widening**: Added Mid-Side (M/S) processing to enhance spatial separation, particularly beneficial for the fallback synth engine.

### 2. Dynamics Processing (`src/focus/dsp/dynamics.py`)
- **True Peak Limiter**: Implemented a transparent brickwall limiter with a -0.1 dBTP ceiling to prevent digital clipping and ensure consistent volume levels across different musical textures.
- **Oversampling**: Uses 2x oversampling for accurate inter-sample peak detection.

### 3. Audio Pipeline & CLI Integration
- **Overlap-Add Processing**: Integrated a crossfade mechanism (`OverlapAddState`) into the audio loop to eliminate micro-clicks at buffer/chunk boundaries.
- **DSP Chain**: Updated `src/focus/cli.py` and `src/focus/audio/pipeline.py` to apply effects in the optimal order: Entrainment -> Spatialization -> Overlap-Add -> Limiting.
- **New CLI Flags**: Exponentiated new features via `focus start` flags:
  - `--reverb / --no-reverb` (Default: ON)
  - `--stereo-width FLOAT` (Default: 1.2)
  - `--limiter / --no-limiter` (Default: ON)

### 4. Documentation & Meta
- **README.md**: Updated with "Audio Enhancements" section and usage examples.
- **GEMINI.md**: Added persistent instructions to always update CLI help and README when adding user-facing features.
- **ROADMAP.md**: Phase 3 items are now ready for check-off.

## Verification
Implemented a comprehensive test suite with **51 passing tests**:
- `tests/test_spatial.py`: Verifies reverb tail energy, stereo image collapse/expansion, and RMS preservation.
- `tests/test_dynamics.py`: Verifies True Peak ceiling enforcement and dB conversion accuracy.
- `tests/test_pipeline_overlap.py`: Ensures glitch-free transitions across multiple chunks.
- Balanced existing entrainment tests.

### Manual Verification
- Verified `--help` output correctly displays new options.
- Tested `focus start --mock --output test.wav` to confirm signal path integrity.

## Checklist
- [x] Implementation matches ROADMAP definition.
- [x] Unit tests cover new DSP modules.
- [x] CLI `--help` updated.
- [x] README.md updated.
- [x] GEMINI.md instructions added.
