"""Environment setup for the physics/time contract suite.

* Forces the dummy SDL driver so the "human" render path runs headlessly.
* Makes ``physics_time_helpers`` importable from every feature sub-directory.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
