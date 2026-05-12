from __future__ import annotations

from pathlib import Path

# Allow `python -m mmsec_eval` from repo root without editable install.
_pkg_dir = Path(__file__).resolve().parent
_src_pkg = _pkg_dir.parent / "src" / "mmsec_eval"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))  # type: ignore[name-defined]

