# Focus Music Generator - Development Roadmap

This document outlines the strategic development plan for the Focus Music Generator, prioritizing audio engineering enhancements and neural entrainment effectiveness.

## Phase 1: Binaural Beat Integration 🎧
**Objective:** Enhance entrainment effectiveness for headphone users by implementing binaural beat generation alongside existing amplitude modulation.

### Concept
Binaural beats utilize the brain's processing of slight frequency differences between the left and right ears to perceive a third "phantom" frequency. This method is often more subtle and effective for sustained focus than simple amplitude modulation (tremolo).

### implementation Steps
- [ ] Create `src/focus/dsp/binaural.py` module.
- [ ] Implement `apply_binaural_entrainment` function:
    - Left Channel: Base frequency
    - Right Channel: Base frequency + Target Entrainment Frequency (e.g., +15Hz)
- [ ] Update `FocusProfile` to support `entrainment_type` (AM vs. Binaural).
- [ ] Add CLI flag `--entrainment-mode` (`am` or `binaural`).
- [ ] Ensure strict stereo separation in the output chain.

### Expected Impact
- Deeper focus states for headphone users.
- Less auditory fatigue compared to constant tremolo effects.
- Better support for ADHD concentration profiles.

---

## Phase 2: Dynamic Adaptive Modulation 🌊
**Objective:** Mimic natural human concentration cycles by adjusting modulation parameters relative to the session duration.

### Concept
Fixed modulation frequencies can lead to mental fatigue. Adaptive modulation varies the entrainment frequency to match the natural "flow" of a work session: warm-up (Alpha), peak focus (Beta), and cool-down (Alpha/Theta).

### Implementation Steps
- [ ] Define `ModulationCurve` data structure in `profiles.py`.
- [ ] Design default curves:
    - **Warm-up:** Start at 10Hz, ramp to target (e.g., 18Hz) over 5 minutes.
    - **Sustained:** Maintain target freq with subtle micro-variations.
    - **Cool-down:** Ramp down to 8-10Hz in the final 5 minutes.
- [ ] Update `_run_session` in `cli.py` to calculate instantaneous frequency based on elapsed time.
- [ ] Interpolate `modulation_freq` and `modulation_depth` in real-time.

### Expected Impact
- Reduced "entrainment fatigue" during long sessions.
- Smoother transitions into and out of deep work states.
- clearer signal to the brain that the session is concluding.

---

## Phase 3: Audio Quality & Spatial Enhancement 🔊
**Objective:** Elevate the listening experience with professional audio processing to ensure long-term listening comfort.

### Concept
Raw generated audio or simple sine-wave modulation can feel "dry" or sterile. Adding spatial effects and safety limiting creates a more polished, comfortable sound field suitable for hours of background listening.

### Implementation Steps
- [ ] **Spatialization:**
    - Implement a subtle reverb algorithm in `src/focus/dsp/spatial.py` (e.g., Schroeder or Freeverb).
    - Add stereo widening for the fallback synth engine.
- [ ] **Dynamics Processing:**
    - Implement a transparent limiter/soft-clipper to prevent digital overs/clipping during modulation peaks.
    - Ensure output never exceeds -0.1 dBTP (True Peak).
- [ ] **Glitch Removal:**
    - Implement "Overlap-Add" processing for chunk transitions to eliminate potential micro-clicks at buffer boundaries.

### Expected Impact
- "Premium" feel to the audio output.
- Reduced listening fatigue due to harsh transients or dry signals.
- Safer audio levels for prolonged headphone use.
