from __future__ import annotations

import sys

from CarlaBEV.cli.run_interactive import main


if __name__ == "__main__":
    raise SystemExit(main(["--reset-mode", "authored", *sys.argv[1:]]))
