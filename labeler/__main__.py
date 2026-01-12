from __future__ import annotations

import sys

import typer

from labeler.cli import run

if len(sys.argv) > 1 and sys.argv[1] == "run":
    sys.argv.pop(1)

typer.run(run)
