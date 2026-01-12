from __future__ import annotations

import sys
from pathlib import Path
from pkgutil import extend_path

_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if _src.exists():
    src_str = str(_src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

__path__ = extend_path(__path__, __name__)
