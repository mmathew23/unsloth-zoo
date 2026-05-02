import sys
import warnings
from pathlib import Path


warnings.filterwarnings("error")
warnings.filterwarnings(
    "ignore",
    message = r"`torch\.jit\.script_method` is deprecated\..*",
    category = DeprecationWarning,
    module = r"torch\.jit\._script",
)
warnings.filterwarnings(
    "ignore",
    message = r"`torch\.jit\.interface` is deprecated\..*",
    category = DeprecationWarning,
    module = r"torch\.jit\._script",
)
warnings.filterwarnings(
    "ignore",
    message = r"Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers\..*",
    category = FutureWarning,
    module = r"transformers\.utils\.hub",
)


def pytest_configure(config):
    config.addinivalue_line("filterwarnings", "error")
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:`torch\.jit\.script_method` is deprecated\..*:DeprecationWarning:torch\.jit\._script",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:`torch\.jit\.interface` is deprecated\..*:DeprecationWarning:torch\.jit\._script",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers\..*:FutureWarning:transformers\.utils\.hub",
    )

WORKSPACE = Path(__file__).resolve().parents[3]
UNSLOTH_ROOT = WORKSPACE / "unsloth"
if str(UNSLOTH_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSLOTH_ROOT))

# Import Unsloth through its normal package entrypoint before importing
# unsloth_zoo modules; unsloth_zoo intentionally rejects standalone imports.
import unsloth  # noqa: F401, E402
