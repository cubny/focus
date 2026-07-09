"""Terminal spectrum visualizer: a classic bar analyzer rising from the bottom.

Bars are drawn with Unicode block glyphs (black & white). Rendering uses the
same in-place multi-row ANSI pattern as :mod:`focus.ui.launcher` (cursor-up +
erase-to-end each frame), and reuses :func:`format_status_line` for the bottom
status row so the status text stays single-sourced.
"""

from __future__ import annotations

import shutil
import sys

import numpy as np

from focus.analysis.realtime import SpectrumAnalyzer
from focus.ui.transport import PlaybackState, format_status_line

# Glyph ramp indexed by eighths of a cell: 0 = empty, 8 = full block.
_GLYPHS = np.array(list(" ▁▂▃▄▅▆▇█"))


class SpectrumRenderer:
    """Pure, testable conversion of band levels into terminal rows.

    Bars rise from the bottom: the last row is densest. Each returned string is a
    full row of ``len(values)`` glyphs; rows are ordered top -> bottom.
    """

    @staticmethod
    def render_rows(values: np.ndarray, height: int) -> list[str]:
        if height < 1:
            return []
        # Filled eighths per column, then per (row, column) how much of that cell
        # is filled. Vectorized so a wide terminal stays cheap at 25+ fps.
        eighths = np.clip(np.asarray(values, dtype=float), 0.0, 1.0) * (height * 8)
        # Row 0 is the top; a row's distance from the bottom scales its threshold.
        from_bottom = np.arange(height - 1, -1, -1)[:, None]
        cell = eighths[None, :] - from_bottom * 8  # (height, num_bands)
        idx = np.clip(np.floor(cell), 0, 8).astype(int)
        grid = _GLYPHS[idx]
        return ["".join(row) for row in grid]


class SpectrumDisplay:
    """Owns the in-place terminal block: spectrum bars plus a status row."""

    def __init__(
        self,
        state: PlaybackState,
        analyzer: SpectrumAnalyzer,
        stream=None,
        height: int = 12,
        fps: int = 25,
        source=None,
    ) -> None:
        self.state = state
        self.analyzer = analyzer
        self.stream = stream or sys.stdout
        self.height = height
        self.fps = fps
        # Callable returning the latest N played mono samples. Pulled every frame
        # so the bars track what's actually heard (not the producer, which runs
        # seconds ahead behind the playback buffer). May be set after construction
        # once the audio output exists.
        self.source = source
        self._active = False
        self._prev_lines = 0

    def start(self) -> None:
        self._active = True
        self._prev_lines = 0
        # Hide the cursor for a flicker-free repaint.
        self.stream.write("\x1b[?25l")
        self.stream.flush()

    def push(self, chunk: np.ndarray) -> None:
        self.analyzer.push(chunk)

    def render(self) -> None:
        if not self._active:
            return
        # Refresh the analysis window from the live playback tap each frame.
        if self.source is not None:
            self.analyzer.push(self.source(self.analyzer.fft_size))
        size = shutil.get_terminal_size(fallback=(80, 24))
        cols = max(1, size.columns)
        lines = max(1, size.lines)

        status = format_status_line(self.state)
        if len(status) > cols - 1:
            status = status[: cols - 1]

        # Very short terminals: fall back to a status-only line.
        if lines < 4:
            rows: list[str] = []
        else:
            h = max(1, min(self.height, lines - 2))
            num_bands = cols
            values = self.analyzer.compute(num_bands)
            rows = SpectrumRenderer.render_rows(values, h)

        block = "\n".join(rows + [status])
        out = ["\x1b[?7l"]  # disable autowrap for the repaint
        if self._prev_lines:
            out.append(f"\x1b[{self._prev_lines}A")
        out.append("\r\x1b[J")  # move to column 0, erase from cursor to end
        out.append(block)
        out.append("\x1b[?7h")  # restore autowrap
        self.stream.write("".join(out))
        self.stream.flush()
        self._prev_lines = len(rows)  # cursor ends on the status row

    def finish(self) -> None:
        if not self._active:
            return
        out = ["\x1b[?7l"]
        if self._prev_lines:
            out.append(f"\x1b[{self._prev_lines}A")
        out.append("\r\x1b[J")  # erase the whole block
        out.append("\x1b[?7h\x1b[?25h")  # restore autowrap + show cursor
        self.stream.write("".join(out))
        self.stream.flush()
        self._active = False
        self._prev_lines = 0
