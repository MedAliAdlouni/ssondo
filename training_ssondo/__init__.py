import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent

DATA = os.environ.get("DATA", str(_PACKAGE_ROOT / "data"))
OUTPUTS = os.environ.get("OUTPUTS", str(_PACKAGE_ROOT / "outputs"))
