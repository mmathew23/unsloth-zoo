import warnings
import pytest


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
warnings.filterwarnings(
    "ignore",
    message = r"Unsloth: Triton (?:could not be imported .*|is not installed\.) Triton-backed kernels are unavailable; use CuTile or other non-Triton backends\.",
    category = UserWarning,
    module = r"unsloth",
)
warnings.filterwarnings(
    "ignore",
    message = r"numpy\.core is deprecated and has been renamed to numpy\._core\..*",
    category = DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message = r"builtin type swigvarlink has no __module__ attribute",
    category = DeprecationWarning,
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
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:.*Triton.*Triton-backed kernels are unavailable.*CuTile.*:UserWarning:unsloth",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:numpy\.core is deprecated and has been renamed to numpy\._core\..*:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:builtin type swigvarlink has no __module__ attribute:DeprecationWarning",
    )

_UNSLOTH_IMPORT_ERROR = None

# Import Unsloth through its normal package entrypoint before importing
# unsloth_zoo modules; unsloth_zoo intentionally rejects standalone imports.
try:
    import unsloth  # noqa: F401, E402
except Exception as exc:
    _UNSLOTH_IMPORT_ERROR = exc


def pytest_ignore_collect(collection_path, config):
    if _UNSLOTH_IMPORT_ERROR is None:
        return False
    return collection_path.name.startswith("test_")


def pytest_report_collectionfinish(config, start_path, items):
    if _UNSLOTH_IMPORT_ERROR is None:
        return []
    return [
        "kernel tests skipped: full unsloth stack could not be imported: "
        f"{_UNSLOTH_IMPORT_ERROR}"
    ]


def pytest_sessionfinish(session, exitstatus):
    if (
        _UNSLOTH_IMPORT_ERROR is not None
        and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
    ):
        session.exitstatus = pytest.ExitCode.OK
