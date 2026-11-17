"""Regenerate the pip requirement files from the locked uv environment.

Server installs use requirements.txt (headless CLI); workstations that also
run the Qt GUI use requirements-gui.txt.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYRADIOMICS_NOTE = """
# pyradiomics 3.0.1 has no CPython 3.11 wheel and cannot be installed with pip.
# Install it manually when feature extraction is needed (see README).
# pyradiomics==3.0.1
"""


def export(target: str, extra: str | None = None) -> None:
    command = [
        "uv",
        "export",
        "--frozen",
        "--no-hashes",
        "--no-emit-project",
        "-o",
        target,
    ]
    if extra:
        command.extend(["--extra", extra])
    subprocess.run(command, cwd=ROOT, check=True)

    path = ROOT / target
    text = path.read_text(encoding="utf-8")
    if "pyradiomics" not in text:
        path.write_text(text + PYRADIOMICS_NOTE, encoding="utf-8")


def main() -> int:
    export("requirements.txt")
    export("requirements-gui.txt", extra="gui")
    print("requirements.txt and requirements-gui.txt updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
