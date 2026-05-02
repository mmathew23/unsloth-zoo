import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
UNSLOTH_ROOT = WORKSPACE / "unsloth"
if str(UNSLOTH_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSLOTH_ROOT))

# Import Unsloth through its normal package entrypoint before importing
# unsloth_zoo modules; unsloth_zoo intentionally rejects standalone imports.
import unsloth  # noqa: F401, E402
