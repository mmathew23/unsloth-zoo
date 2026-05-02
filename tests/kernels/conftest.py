import warnings


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

# Import Unsloth through its normal package entrypoint before importing
# unsloth_zoo modules; unsloth_zoo intentionally rejects standalone imports.
import unsloth  # noqa: F401, E402
