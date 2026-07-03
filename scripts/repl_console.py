"""Render Python REPL transcripts deterministically.

Drivers create a Session and feed source strings via `run()`. The Session
executes them through a stdlib `code.InteractiveConsole`, capturing input,
output, and tracebacks in the canonical interactive-Python format
(>>> prompts, ... continuations, standard tracebacks).
"""

from __future__ import annotations

import code
import contextlib
import io


class Session:
    """A scriptable Python REPL session that renders as a >>>-prefixed transcript."""

    def __init__(self) -> None:
        self._console = code.InteractiveConsole(locals={})
        self._lines: list[str] = []

    def run(self, source: str) -> None:
        """Execute `source` (one logical block) and append its rendering."""
        self._lines.extend(self._format_input(source))

        out_buf = io.StringIO()
        # console.write is what InteractiveConsole uses for tracebacks
        orig_write = self._console.write
        self._console.write = out_buf.write
        try:
            with contextlib.redirect_stdout(out_buf):
                src_lines = source.split("\n")
                more = False
                for line in src_lines:
                    more = self._console.push(line)
                # Force completion if the console is still waiting
                if more:
                    self._console.push("")
        finally:
            self._console.write = orig_write

        output = out_buf.getvalue()
        if output:
            # Strip a single trailing newline; we'll add one when joining
            self._lines.append(output.rstrip("\n"))

    def render(self) -> str:
        """Return the full transcript as a single string ending with newline."""
        return "\n".join(self._lines) + "\n"

    @staticmethod
    def _format_input(source: str) -> list[str]:
        src_lines = source.split("\n")
        return [">>> " + src_lines[0]] + ["... " + line for line in src_lines[1:]]
