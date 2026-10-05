"""`python -m tradex.options check [dir]`: schema-check options strategy files (default: the bundled specs)."""
from __future__ import annotations

import sys
from pathlib import Path

from tradex.options.spec import load_dir

DEFAULT_DIR = Path(__file__).resolve().parent / "specs"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "check":
        print(__doc__)
        return 2
    path = Path(args[1]) if len(args) > 1 else DEFAULT_DIR
    bad = 0
    for spec in load_dir(path):
        errs = spec.validate()
        print(f"{'FAIL' if errs else 'ok  '} {spec.id}")
        for e in errs:
            print(f"     - {e}")
        bad += bool(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
