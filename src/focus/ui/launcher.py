"""Interactive profile picker shown when ``focus`` is run bare in a terminal.

Makes the tool usable with zero knowledge of profile names or flags: arrow keys
or number keys to choose, Enter to start, ``q``/Esc to cancel. Power-user flags
remain available via ``focus start ...``.
"""

import os
import sys

import click

from focus.profiles import FocusProfile, list_profiles


def _getch(fd: int | None = None) -> bytes:
    """Read one keypress (or escape sequence) in cbreak mode.

    Reads the raw fd directly (not a buffered reader): under cbreak with VMIN=1
    a single ``os.read`` returns a whole escape burst (e.g. ``\\x1b[A`` for an
    arrow key) in one call, so arrow keys are not mistaken for a bare Escape.
    """
    import termios
    import tty

    fd = sys.stdin.fileno() if fd is None else fd
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return os.read(fd, 6)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _menu_lines(profiles: list[FocusProfile], selected: int) -> list[str]:
    """Build the menu as a list of single (unwrapped) lines.

    Returning exact lines lets the redraw move the cursor up by a precise count;
    no embedded newlines means ``len(lines)`` is the true rendered height.
    """
    lines = ["", "🎧  " + click.style("Choose a focus profile", bold=True), ""]
    for i, p in enumerate(profiles):
        marker = click.style("❯", fg="cyan", bold=True) if i == selected else " "
        name = click.style(p.name, fg="cyan", bold=(i == selected))
        lines.append(f"  {marker} {i + 1}. {name}")
        lines.append(f"      {p.description}")
        lines.append(f"      {p.modulation_freq:.0f} Hz @ {p.modulation_depth:.0%}")
    lines.append("")
    lines.append("  ↑↓ move · 1-9 jump · Enter start · q cancel")
    return lines


def run_launcher() -> str | None:
    """Show the picker and return the chosen profile name, or None if cancelled.

    Caller is responsible for ensuring stdin is a TTY.
    """
    profiles = list_profiles()
    if not profiles:
        return None

    selected = 0
    prev_lines = 0
    out = sys.stdout
    try:
        while True:
            lines = _menu_lines(profiles, selected)
            if prev_lines:
                # Return to the top of the previous block and clear everything
                # below it, so nothing from the prior frame can ghost through.
                out.write(f"\x1b[{prev_lines}A")
            out.write("\x1b[J")
            out.write("\n".join(lines) + "\n")
            out.flush()
            prev_lines = len(lines)

            key = _getch()
            if key in (b"\r", b"\n"):
                return profiles[selected].name
            if key in (b"q", b"Q", b"\x1b"):
                return None
            if key == b"\x1b[A":  # up
                selected = (selected - 1) % len(profiles)
            elif key == b"\x1b[B":  # down
                selected = (selected + 1) % len(profiles)
            elif key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(profiles):
                    selected = idx
    finally:
        out.write("\n")
        out.flush()
