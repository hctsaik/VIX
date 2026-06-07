"""Enable `python -m vix ...` (besides the installed `vix` console script) — useful in
air-gapped / no-PATH environments where the entry-point script isn't on PATH."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
