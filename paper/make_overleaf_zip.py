"""Build paper/overleaf_upload.zip: everything Overleaf needs to compile the paper, nothing else.

    python3 paper/make_overleaf_zip.py

Upload on Overleaf with "New Project" -> "Upload Project"; compiler pdfLaTeX, main document main.tex.
Regenerate figures (.venv/bin/python paper/figures/make_figures.py) and tables (python3 paper/figures/make_tables.py)
first if the results changed.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
OUT = PAPER / "overleaf_upload.zip"
FIXED = ["main.tex", "references.bib", "acl.sty", "acl_natbib.bst"]


def inputs(tex: Path) -> list[Path]:
    """Files reached from ``tex`` through \\input and \\includegraphics (recursively), relative to the paper root."""
    text = re.sub(r"(?<!\\)%.*", "", tex.read_text())
    found = []
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        path = PAPER / (name if name.endswith(".tex") else name + ".tex")
        found += [path] + inputs(path)
    found += [PAPER / g for g in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)]
    return found


def main() -> None:
    files = [PAPER / f for f in FIXED] + inputs(PAPER / "main.tex")
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(set(files)):
            z.write(f, f.relative_to(PAPER).as_posix())
    print(f"{OUT} ({OUT.stat().st_size // 1024} KB, {len(set(files))} files)")
    for f in sorted(set(files)):
        print("  ", f.relative_to(PAPER).as_posix())


if __name__ == "__main__":
    main()
