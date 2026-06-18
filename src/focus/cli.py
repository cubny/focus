"""Command-line interface for Focus music generator."""

import asyncio
import sys

import click

from focus.dsp.dynamics import LimiterState, apply_limiter
from focus.dsp.entrainment import ModulationState, apply_entrainment, apply_fade_out
from focus.dsp.spatial import ReverbState, apply_reverb, apply_stereo_widening
from focus.generation.lyria_client import LyriaConfig, create_client
from focus.profiles import FocusProfile, get_profile, list_profiles
from focus.ui.transport import KeyboardController, PlaybackState, StatusLine

# Check for optional dependencies
try:
    import numpy as np
    import sounddevice as sd

    AUDIO_AVAILABLE = True
except (ImportError, OSError):
    np = None
    sd = None
    AUDIO_AVAILABLE = False


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0")
@click.pass_context
def main(ctx):
    """Focus - Neural entrainment music generator.

    Generate focus-enhancing music using AI (Google Lyria) with
    neural entrainment modulation for improved concentration.

    Run `focus` with no arguments in a terminal to pick a profile interactively.

    Quick Usage:\n
        focus start --profile deep-work \n
        focus start --duration 600  # 10 minute session \n
        focus start --output session.wav \n
    """
    if ctx.invoked_subcommand is not None:
        return

    # Bare invocation: drop into the interactive picker when attached to a
    # terminal; otherwise (pipes, CI) fall back to the usual help text.
    if sys.stdin.isatty() and sys.stdout.isatty():
        from focus.ui.launcher import run_launcher

        choice = run_launcher()
        if choice:
            launch_session(profile=choice)
    else:
        click.echo(ctx.get_help())


@main.command("profiles")
def show_profiles():
    """List available focus profiles."""
    profiles = list_profiles()

    click.echo("\n🎧 Available Focus Profiles\n")
    click.echo("-" * 60)

    for p in profiles:
        click.echo(f"\n  {click.style(p.name, fg='cyan', bold=True)}")
        click.echo(f"  {p.description}")
        click.echo(f"  Modulation: {p.modulation_freq:.0f} Hz @ {p.modulation_depth:.0%} depth")

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
@click.option(
    "--track-duration",
    type=int,
    default=9,
    help="Duration of each track before rotation (1-9 minutes, default: 9)",
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
    track_duration: int,
):
    """Start a focus music session.

    Examples:

        focus start --profile deep-work

        focus start -p light-study --duration 300

        focus start --frequency 16 --depth 0.3 --mock
    """
    launch_session(
        profile=profile,
        frequency=frequency,
        depth=depth,
        prompt=prompt,
        mock=mock,
        duration=duration,
        output=output,
        reverb=reverb,
        stereo_width=stereo_width,
        limiter=limiter,
        verbose=verbose,
        track_duration=track_duration,
    )


def launch_session(
    profile: str,
    frequency: float | None = None,
    depth: float | None = None,
    prompt: str | None = None,
    mock: bool = False,
    duration: int | None = None,
    output: str | None = None,
    reverb: bool = True,
    stereo_width: float = 1.2,
    limiter: bool = True,
    verbose: bool = False,
    track_duration: int = 9,
):
    """Resolve a profile, apply overrides, and run a session.

    Shared entry point for both the ``start`` command and the bare-``focus``
    interactive launcher.
    """
    if duration is not None and duration < 60:
        click.echo(
            "Error: Duration must be at least 60 seconds to allow for intro/outro phases.",
            err=True,
        )
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
    click.echo(
        f"   Modulation: {focus_profile.modulation_freq:.0f} Hz @ "
        f"{focus_profile.modulation_depth:.0%}"
    )
    click.echo(
        f"   Effects: Reverb={'ON' if reverb else 'OFF'}, "
        f"Width={stereo_width}, Limiter={'ON' if limiter else 'OFF'}"
    )
    if verbose:
        click.echo(
            f"   BPM: {focus_profile.bpm}, Density: {focus_profile.density}, "
            f"Brightness: {focus_profile.brightness}"
        )
        click.echo(f"   Prompt: {focus_profile.prompt[:60]}...")
    if duration:
        click.echo(f"   Duration: {duration} seconds")
    if output:
        click.echo(f"   Output: {output}")
    if sys.stdin.isatty() and sys.stdout.isatty():
        click.echo("\n   Controls: [space] pause  [n] next take  [↑↓] volume  [?] help  [q] quit\n")
    else:
        click.echo("\n   Press Ctrl+C to stop\n")

    try:
        asyncio.run(
            _run_session(
                focus_profile,
                mock,
                duration,
                output,
                reverb=reverb,
                stereo_width=stereo_width,
                limiter=limiter,
                verbose=verbose,
                track_duration=track_duration,
            )
        )
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
    verbose: bool = False,
    track_duration: int = 9,
):
    """Run the audio generation session."""
    try:
        from focus.audio.output import AudioOutput, FileAudioOutput
    except ImportError:
        if not use_mock:
            click.echo("Error: sounddevice not available", err=True)
            return

    sample_rate = 48000

    # Clamp track duration to valid range (1-9 minutes)
    track_duration_seconds = max(60, min(9 * 60, track_duration * 60))

    def build_config(phase: str) -> LyriaConfig:
        """Build a Lyria config for the given musical phase."""
        bpm = profile.bpm or 120
        density = profile.density or 0.5
        brightness = profile.brightness or 0.5
        if phase == "intro" and profile.intro_prompt:
            return LyriaConfig(
                prompt=f"{profile.intro_prompt}, {profile.prompt}",
                bpm=bpm,
                density=max(0.1, density - 0.2),  # Start with lower density
                brightness=brightness,
            )
        if phase == "outro" and profile.outro_prompt:
            return LyriaConfig(
                prompt=f"{profile.outro_prompt}, {profile.prompt}",
                bpm=bpm,
                density=density,
                brightness=brightness,
            )
        return LyriaConfig(prompt=profile.prompt, bpm=bpm, density=density, brightness=brightness)

    def make_client(phase: str):
        return create_client(
            build_config(phase),
            use_mock=use_mock,
            verbose=verbose,
            session_duration=track_duration_seconds,
        )

    if not AUDIO_AVAILABLE:
        client = make_client("main")
        click.echo("⚠️  sounddevice not available, running in test mode")
        await client.connect()
        chunk_count = 0
        async for chunk in client.generate_stream():
            chunk_count += 1
            click.echo(
                f"  📦 Chunk {chunk_count}: shape={chunk.shape}, "
                f"range=[{chunk.min():.2f}, {chunk.max():.2f}]"
            )
            if chunk_count >= 10:
                break
        await client.stop()
        return

    # Interactive transport controls (only attached to a real terminal)
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    state = None
    keyboard = None
    status_line = None
    if interactive:
        state = PlaybackState(
            profile_name=profile.name,
            modulation_freq=profile.modulation_freq,
            modulation_depth=profile.modulation_depth,
            status="connecting",
        )
        if not verbose:
            # The live status line and -v logging both want the bottom line;
            # when verbose, the logs already convey state, so skip the line.
            status_line = StatusLine()
            status_line.start()
            status_line.render(state)

        # Redraw immediately on each keypress so pause / volume / help toggles
        # are reflected instantly instead of on the next audio chunk.
        def _on_key():
            if status_line is not None:
                status_line.render(state)

        keyboard = KeyboardController(state, on_change=_on_key)
        try:
            keyboard.start()
        except Exception:
            keyboard.stop()
            if status_line is not None:
                status_line.finish()
            raise

    # Initialize DSP state
    mod_state = ModulationState()
    reverb_state = ReverbState() if reverb else None
    limiter_state = LimiterState(ceiling_linear=0.989) if limiter else None  # -0.1 dBTP

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

    if verbose:
        click.echo(f"   🔊 Audio device: {sd.query_devices(sd.default.device[1])['name']}")
        if duration:
            click.echo(
                f"   🎵 Musical phases: intro ({intro_duration}s) → "
                f"main → outro ({outro_duration}s)"
            )

    # Connect to generator
    client = make_client(current_phase)
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

    session_complete = False

    try:
        # Outer loop: each iteration consumes one generator until it ends or an
        # interactive control (pause / next take) asks us to reconnect.
        while not session_complete:
            stream = client.generate_stream()
            reconnect = False

            async for chunk in stream:
                chunk_count += 1
                chunk_seconds = len(chunk) / sample_rate
                total_seconds += chunk_seconds

                if verbose:
                    # Log amplitude to verify signal presence
                    max_amp = np.max(np.abs(chunk))
                    click.echo(
                        f"   📦 Chunk {chunk_count}: {len(chunk)} samples, "
                        f"max_amp={max_amp:.3f}, phase={current_phase}"
                    )

                # Phase transitions for timed sessions
                if duration:
                    # Transition: intro -> main (after intro_duration)
                    if (
                        current_phase == "intro"
                        and total_seconds >= intro_duration
                        and not phase_switched_to_main
                    ):
                        current_phase = "main"
                        phase_switched_to_main = True
                        await client.set_prompt(profile.prompt)
                        if verbose:
                            click.echo("   🎵 Phase transition: intro → main")

                    # Transition: main -> outro (outro_duration before end)
                    time_remaining = duration - total_seconds
                    if (
                        current_phase == "main"
                        and time_remaining <= outro_duration
                        and not phase_switched_to_outro
                    ):
                        if profile.outro_prompt:
                            current_phase = "outro"
                            phase_switched_to_outro = True
                            outro_full_prompt = f"{profile.outro_prompt}, {profile.prompt}"
                            await client.set_prompt(outro_full_prompt)
                            if verbose:
                                click.echo("   🎵 Phase transition: main → outro")

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
                        fade_progress = 1.0 - (
                            fade_in_samples_remaining / (fade_duration * sample_rate)
                        )
                        end_progress = fade_progress + chunk_samples / (fade_duration * sample_rate)
                        t = np.linspace(
                            fade_progress * np.pi / 2,
                            end_progress * np.pi / 2,
                            chunk_samples,
                        )
                        envelope = np.sin(t) ** 2
                        if modulated.ndim == 2:
                            modulated = modulated * envelope[:, np.newaxis]
                        else:
                            modulated = modulated * envelope
                        modulated = modulated.astype(np.float32)
                    else:
                        # Partial fade on this chunk
                        fade_progress = 1.0 - (
                            fade_in_samples_remaining / (fade_duration * sample_rate)
                        )
                        t = np.linspace(
                            fade_progress * np.pi / 2,
                            np.pi / 2,
                            fade_in_samples_remaining,
                        )
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
                        modulated, sample_rate, state=reverb_state
                    )

                # 2. Stereo Widening
                if abs(stereo_width - 1.0) > 0.01:
                    modulated = apply_stereo_widening(modulated, width=stereo_width)

                # 4. Dynamics (Limiter)
                if limiter:
                    modulated, limiter_state = apply_limiter(
                        modulated, sample_rate, state=limiter_state
                    )

                # Output gain (volume) is applied inside AudioOutput, post-limiter
                if state is not None:
                    output.set_volume(state.volume)

                # ALWAYS write to real-time output immediately (no buffering delay)
                output.write(modulated)

                # For file output with duration: buffer the last 5 seconds for fade-out
                if file_output:
                    if duration:
                        fade_out_buffer.append(modulated.copy())
                        # Keep only enough buffer for fade-out duration
                        total_buffered = sum(len(c) for c in fade_out_buffer)
                        fade_out_samples = int(fade_duration * sample_rate)
                        while total_buffered > fade_out_samples and len(fade_out_buffer) > 1:
                            old_chunk = fade_out_buffer.pop(0)
                            total_buffered -= len(old_chunk)
                            # Write the old chunk that's no longer in fade zone
                            file_output.write(old_chunk)
                    else:
                        # No duration limit, write immediately to file
                        file_output.write(modulated)

                # Check duration limit
                if duration and total_seconds >= duration:
                    if verbose:
                        click.echo(f"\n   ⏱️  Duration reached ({total_seconds:.1f}s)")
                    # Apply fade-out to file output's buffered chunks
                    if fade_out_buffer and file_output:
                        combined = np.concatenate(fade_out_buffer, axis=0)
                        faded = apply_fade_out(combined, sample_rate, fade_duration)
                        file_output.write(faded)
                        fade_out_buffer.clear()
                    session_complete = True
                    break

                # Interactive controls
                if state is not None:
                    state.elapsed_seconds = total_seconds
                    state.buffer_seconds = output.buffer_seconds
                    state.status = "playing"
                    if status_line is not None:
                        status_line.render(state)
                    if state.quit_requested:
                        session_complete = True
                        break
                    if state.paused or state.skip_requested:
                        reconnect = True
                        break

                # Yield control to event loop to keep UI responsive
                await asyncio.sleep(0)

            # Close the abandoned/finished generator before reconnecting
            try:
                await stream.aclose()
            except Exception:
                pass

            # End the session unless an interactive control asked to reconnect
            if session_complete or state is None or not reconnect:
                break

            # Pause: tear the session down (stops burning quota), wait, reconnect
            if state.paused:
                output.pause()
                await client.stop()
                state.status = "paused"
                if status_line is not None:
                    status_line.render(state)
                while state.paused and not state.quit_requested:
                    await asyncio.sleep(0.15)
                    if status_line is not None:
                        status_line.render(state)
                if state.quit_requested:
                    break
                output.resume()

            # "Next take": force a fresh generation (same profile/phase)
            state.skip_requested = False
            state.status = "reconnecting"
            if status_line is not None:
                status_line.render(state)
            await client.stop()
            client = make_client(current_phase)
            await client.connect()
            # Fade the new take in to avoid a hard join
            fade_in_samples_remaining = int(fade_duration * sample_rate)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        if verbose:
            click.echo(f"   ⚠️  Session error: {e}")
            import traceback

            traceback.print_exc()
    finally:
        # Restore the terminal before any further output
        if keyboard is not None:
            keyboard.stop()
        if status_line is not None:
            status_line.finish()
        # Flush any remaining buffered audio to file (for Ctrl+C case with duration set)
        if fade_out_buffer and file_output:
            combined = np.concatenate(fade_out_buffer, axis=0)
            faded = apply_fade_out(
                combined, sample_rate, min(fade_duration, len(combined) / sample_rate)
            )
            file_output.write(faded)
        output.flush()  # Flush leftover samples with fade-out
        output.stop()
        if file_output:
            file_output.stop()
        await client.stop()
        if verbose:
            buffer_info = f", buffer={output.buffer_seconds:.1f}s"
            underrun_msg = (
                f", {output.underrun_count} underruns" if output.underrun_count > 0 else ""
            )
            click.echo(
                f"   ✓ Session ended after {total_seconds:.1f}s "
                f"({chunk_count} chunks{underrun_msg}{buffer_info})"
            )


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
        import numpy as np
        from scipy.io import wavfile
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
