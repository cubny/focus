"""Command-line interface for Focus music generator."""

import asyncio
import click
import sys
import os

from focus.profiles import get_profile, list_profiles, FocusProfile
from focus.generation.lyria_client import LyriaConfig, create_client
from focus.dsp.entrainment import (
    ModulationState, 
    apply_entrainment, 
    apply_fade_in, 
    apply_fade_out
)
from focus.dsp.spatial import (
    ReverbState,
    apply_reverb,
    apply_stereo_widening
)
from focus.dsp.dynamics import (
    LimiterState,
    apply_limiter
)
from focus.audio.pipeline import OverlapAddState

# Check for optional dependencies
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Focus - Neural entrainment music generator.

    Generate focus-enhancing music using AI (Google Lyria) with
    neural entrainment modulation for improved concentration.

    Quick Usage:\n
        focus start --profile deep-work \n
        focus start --duration 600  # 10 minute session \n
        focus start --output session.wav \n
    """
    pass


@main.command("profiles")
def show_profiles():
    """List available focus profiles."""
    profiles = list_profiles()

    click.echo("\n🎧 Available Focus Profiles\n")
    click.echo("-" * 60)

    for p in profiles:
        click.echo(f"\n  {click.style(p.name, fg='cyan', bold=True)}")
        click.echo(f"  {p.description}")
        click.echo(
            f"  Modulation: {p.modulation_freq:.0f} Hz @ {p.modulation_depth:.0%} depth"
        )

    click.echo("\n" + "-" * 60)
    click.echo("\nUsage: focus start --profile <name>\n")


@main.command("start")
@click.option(
    "--profile",
    "-p",
    type=str,
    default="deep-work",
    help="Focus profile to use (see 'focus profiles' for list)",
)
@click.option(
    "--frequency",
    "-f",
    type=float,
    default=None,
    help="Override modulation frequency (Hz)",
)
@click.option(
    "--depth",
    "-d",
    type=float,
    default=None,
    help="Override modulation depth (0.0-1.0)",
)
@click.option(
    "--prompt",
    type=str,
    default=None,
    help="Custom Lyria prompt (overrides profile)",
)
@click.option(
    "--mock",
    is_flag=True,
    default=False,
    help="Use mock audio generator (no API key needed)",
)
@click.option(
    "--duration",
    type=int,
    default=None,
    help="Session duration in seconds (minimum: 60, default: unlimited)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save audio to WAV file (in addition to playback)",
)
@click.option(
    "--reverb/--no-reverb",
    is_flag=True,
    default=True,
    help="Enable/disable spatial reverb (default: enabled)",
)
@click.option(
    "--stereo-width",
    type=float,
    default=1.2,
    help="Stereo width enhancement (1.0=original, >1.0=wider)",
)
@click.option(
    "--limiter/--no-limiter",
    is_flag=True,
    default=True,
    help="Enable/disable safety limiter (default: enabled)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed debug output",
)
def start_session(
    profile: str,
    frequency: float | None,
    depth: float | None,
    prompt: str | None,
    mock: bool,
    duration: int | None,
    output: str | None,
    reverb: bool,
    stereo_width: float,
    limiter: bool,
    verbose: bool,
):
    """Start a focus music session.

    Examples:

        focus start --profile deep-work

        focus start -p light-study --duration 300

        focus start --frequency 16 --depth 0.3 --mock
    """
    if duration is not None and duration < 60:
        click.echo("Error: Duration must be at least 60 seconds to allow for intro/outro phases.", err=True)
        sys.exit(1)

    try:
        focus_profile = get_profile(profile)
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Apply overrides
    if frequency or depth or prompt:
        focus_profile = FocusProfile(
            name=focus_profile.name,
            description=focus_profile.description,
            prompt=prompt or focus_profile.prompt,
            modulation_freq=frequency or focus_profile.modulation_freq,
            modulation_depth=depth or focus_profile.modulation_depth,
            bpm=focus_profile.bpm,
            density=focus_profile.density,
            brightness=focus_profile.brightness,
        )

    click.echo(f"\n🎯 Starting focus session: {click.style(focus_profile.name, bold=True)}")
    click.echo(f"   Modulation: {focus_profile.modulation_freq:.0f} Hz @ {focus_profile.modulation_depth:.0%}")
    click.echo(f"   Effects: Reverb={'ON' if reverb else 'OFF'}, Width={stereo_width}, Limiter={'ON' if limiter else 'OFF'}")
    if verbose:
        click.echo(f"   BPM: {focus_profile.bpm}, Density: {focus_profile.density}, Brightness: {focus_profile.brightness}")
        click.echo(f"   Prompt: {focus_profile.prompt[:60]}...")
    if duration:
        click.echo(f"   Duration: {duration} seconds")
    if output:
        click.echo(f"   Output: {output}")
    click.echo("\n   Press Ctrl+C to stop\n")

    try:
        asyncio.run(_run_session(
            focus_profile, 
            mock, 
            duration, 
            output, 
            reverb=reverb,
            stereo_width=stereo_width,
            limiter=limiter,
            verbose=verbose
        ))
    except KeyboardInterrupt:
        click.echo("\n\n🛑 Session stopped by user")
    
    if output:
        click.echo(f"💾 Audio saved to: {output}")
    click.echo("👋 Session ended. Stay focused!\n")


async def _run_session(
    profile: FocusProfile, 
    use_mock: bool, 
    duration: int | None, 
    output_path: str | None = None,
    reverb: bool = True,
    stereo_width: float = 1.2,
    limiter: bool = True,
    verbose: bool = False
):
    """Run the audio generation session."""
    try:
        from focus.audio.output import AudioOutput, FileAudioOutput
    except ImportError:
        if not use_mock:
            click.echo("Error: sounddevice not available", err=True)
            return

    config = LyriaConfig(
        prompt=profile.prompt,
        bpm=profile.bpm or 120,
        density=profile.density or 0.5,
        brightness=profile.brightness or 0.5,
    )

    client = create_client(config, use_mock=use_mock, verbose=verbose)

    if not AUDIO_AVAILABLE:
        click.echo("⚠️  sounddevice not available, running in test mode")
        await client.connect()
        chunk_count = 0
        async for chunk in client.generate_stream():
            chunk_count += 1
            click.echo(f"  📦 Chunk {chunk_count}: shape={chunk.shape}, range=[{chunk.min():.2f}, {chunk.max():.2f}]")
            if chunk_count >= 10:
                break
        await client.stop()
        return

    # Initialize state
    mod_state = ModulationState()
    reverb_state = ReverbState() if reverb else None
    limiter_state = LimiterState(ceiling_linear=0.989) if limiter else None  # -0.1 dBTP
    overlap_state = OverlapAddState()
    
    sample_rate = 48000
    chunk_count = 0
    total_seconds = 0.0
    
    # Fade settings (in seconds)
    fade_duration = 5.0
    fade_in_samples_remaining = int(fade_duration * sample_rate)
    fade_out_buffer = []  # Buffer for fade-out when duration is set

    # Phase management for timed sessions (natural musical evolution)
    # Phases: intro -> main -> outro
    intro_duration = 15.0  # seconds for buildup phase
    outro_duration = 30.0  # seconds for wind-down phase
    current_phase = "intro" if duration and profile.intro_prompt else "main"
    phase_switched_to_main = current_phase == "main"
    phase_switched_to_outro = False

    # Build initial prompt with intro modifier if timed session
    if duration and profile.intro_prompt and current_phase == "intro":
        initial_prompt = f"{profile.intro_prompt}, {profile.prompt}"
        config = LyriaConfig(
            prompt=initial_prompt,
            bpm=profile.bpm or 120,
            density=max(0.1, (profile.density or 0.5) - 0.2),  # Start with lower density
            brightness=profile.brightness or 0.5,
        )
        client = create_client(config, use_mock=use_mock, verbose=verbose)

    if verbose:
        click.echo(f"   🔊 Audio device: {sd.query_devices(sd.default.device[1])['name']}")
        if duration:
            click.echo(f"   🎵 Musical phases: intro ({intro_duration}s) → main → outro ({outro_duration}s)")

    # Connect to generator
    await client.connect()
    if verbose:
        click.echo("   ✓ Connected to audio generator")

    # Use queue-based output for robust playback
    output = AudioOutput(sample_rate=sample_rate)
    output.start()
    
    # Optional file output
    file_output = None
    if output_path:
        file_output = FileAudioOutput(filepath=output_path, sample_rate=sample_rate)
        file_output.start()

    try:
        async for chunk in client.generate_stream():
            chunk_count += 1
            chunk_seconds = len(chunk) / sample_rate
            total_seconds += chunk_seconds

            if verbose:
                # Log amplitude to verify signal presence
                max_amp = np.max(np.abs(chunk))
                click.echo(f"   📦 Chunk {chunk_count}: {len(chunk)} samples, max_amp={max_amp:.3f}, phase={current_phase}")

            # Phase transitions for timed sessions
            if duration:
                # Transition: intro -> main (after intro_duration)
                if current_phase == "intro" and total_seconds >= intro_duration and not phase_switched_to_main:
                    current_phase = "main"
                    phase_switched_to_main = True
                    await client.set_prompt(profile.prompt)
                    if verbose:
                        click.echo(f"   🎵 Phase transition: intro → main")

                # Transition: main -> outro (outro_duration before end)
                time_remaining = duration - total_seconds
                if current_phase == "main" and time_remaining <= outro_duration and not phase_switched_to_outro:
                    if profile.outro_prompt:
                        current_phase = "outro"
                        phase_switched_to_outro = True
                        outro_full_prompt = f"{profile.outro_prompt}, {profile.prompt}"
                        await client.set_prompt(outro_full_prompt)
                        if verbose:
                            click.echo(f"   🎵 Phase transition: main → outro")

            # Apply neural entrainment
            modulated, mod_state = apply_entrainment(
                chunk,
                sample_rate,
                target_freq=profile.modulation_freq,
                depth=profile.modulation_depth,
                state=mod_state,
            )

            # Apply fade-in to early chunks
            if fade_in_samples_remaining > 0:
                chunk_samples = len(modulated)
                if fade_in_samples_remaining >= chunk_samples:
                    # This entire chunk needs fading
                    fade_progress = 1.0 - (fade_in_samples_remaining / (fade_duration * sample_rate))
                    t = np.linspace(fade_progress * np.pi / 2, (fade_progress + chunk_samples / (fade_duration * sample_rate)) * np.pi / 2, chunk_samples)
                    envelope = np.sin(t) ** 2
                    if modulated.ndim == 2:
                        modulated = modulated * envelope[:, np.newaxis]
                    else:
                        modulated = modulated * envelope
                    modulated = modulated.astype(np.float32)
                else:
                    # Partial fade on this chunk
                    fade_progress = 1.0 - (fade_in_samples_remaining / (fade_duration * sample_rate))
                    t = np.linspace(fade_progress * np.pi / 2, np.pi / 2, fade_in_samples_remaining)
                    envelope = np.sin(t) ** 2
                    if modulated.ndim == 2:
                        modulated[:fade_in_samples_remaining] *= envelope[:, np.newaxis]
                    else:
                        modulated[:fade_in_samples_remaining] *= envelope
                    modulated = modulated.astype(np.float32)
                fade_in_samples_remaining -= chunk_samples

            # --- Phase 3 DSP Chain ---
            
            # 1. Spatialization (Reverb)
            if reverb:
                modulated, reverb_state = apply_reverb(
                    modulated,
                    sample_rate,
                    state=reverb_state
                )

            # 2. Stereo Widening
            if abs(stereo_width - 1.0) > 0.01:
                modulated = apply_stereo_widening(modulated, width=stereo_width)
            
            # 3. Glitch Removal (Overlap-Add)
            modulated = overlap_state.process(modulated)
            
            # 4. Dynamics (Limiter)
            if limiter:
                modulated, limiter_state = apply_limiter(
                    modulated,
                    sample_rate,
                    state=limiter_state
                )

            # Buffer for fade-out when duration is set
            if duration:
                fade_out_buffer.append(modulated.copy())
                # Keep only enough buffer for fade-out duration
                total_buffered = sum(len(c) for c in fade_out_buffer)
                fade_out_samples = int(fade_duration * sample_rate)
                while total_buffered > fade_out_samples and len(fade_out_buffer) > 1:
                    old_chunk = fade_out_buffer.pop(0)
                    total_buffered -= len(old_chunk)
                    # Write the old chunk that's no longer in fade zone
                    output.write(old_chunk)
                    if file_output:
                        file_output.write(old_chunk)
            else:
                # No duration limit, write immediately
                output.write(modulated)
                if file_output:
                    file_output.write(modulated)

            # Check duration limit
            if duration and total_seconds >= duration:
                if verbose:
                    click.echo(f"\n   ⏱️  Duration reached ({total_seconds:.1f}s)")
                # Apply fade-out to buffered chunks
                if fade_out_buffer:
                    combined = np.concatenate(fade_out_buffer, axis=0)
                    faded = apply_fade_out(combined, sample_rate, fade_duration)
                    output.write(faded)
                    if file_output:
                        file_output.write(faded)
                    fade_out_buffer.clear()
                break
                
            # Yield control to event loop to keep UI responsive
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        if verbose:
            click.echo(f"   ⚠️  Session error: {e}")
            import traceback
            traceback.print_exc()
    finally:
        # Flush any remaining buffered audio (for Ctrl+C case with duration set)
        if fade_out_buffer:
            combined = np.concatenate(fade_out_buffer, axis=0)
            faded = apply_fade_out(combined, sample_rate, min(fade_duration, len(combined) / sample_rate))
            output.write(faded)
            if file_output:
                file_output.write(faded)
        output.stop()
        if file_output:
            file_output.stop()
        await client.stop()
        if verbose:
            click.echo(f"   ✓ Session ended after {total_seconds:.1f}s ({chunk_count} chunks)")


@main.command("test-audio")
def test_audio():
    """Test audio output with a simple tone."""
    if not AUDIO_AVAILABLE:
        click.echo("Error: sounddevice not available", err=True)
        sys.exit(1)

    click.echo("\n🔊 Testing audio output...")
    click.echo(f"   Default output: {sd.query_devices(sd.default.device[1])['name']}")

    # Generate a 440 Hz test tone
    sample_rate = 48000
    duration = 2.0
    t = np.arange(int(duration * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    stereo = np.column_stack([tone, tone]).astype(np.float32)

    click.echo("   Playing 440 Hz tone for 2 seconds...")
    try:
        sd.play(stereo, samplerate=sample_rate, blocking=True)
        click.echo("   ✓ Audio test complete!")
    except Exception as e:
        click.echo(f"   ✗ Audio error: {e}", err=True)


@main.command("analyze")
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--expected-freq",
    "-f",
    type=float,
    default=15.0,
    help="Expected modulation frequency (Hz)",
)
@click.option(
    "--expected-depth",
    "-d",
    type=float,
    default=0.3,
    help="Expected modulation depth",
)
def analyze_audio(audio_file: str, expected_freq: float, expected_depth: float):
    """Analyze an audio file for neural entrainment modulation.

    Verifies that amplitude modulation is present at the expected frequency.
    """
    try:
        from scipy.io import wavfile
        import numpy as np
    except ImportError:
        click.echo("Error: scipy is required for audio analysis", err=True)
        sys.exit(1)

    from focus.analysis.fft import generate_report

    click.echo(f"\n📊 Analyzing: {audio_file}\n")

    sample_rate, audio = wavfile.read(audio_file)

    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    report = generate_report(audio, sample_rate, expected_freq, expected_depth)
    click.echo(report)


if __name__ == "__main__":
    main()
