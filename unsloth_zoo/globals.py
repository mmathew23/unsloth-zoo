import os
from pathlib import Path
from filelock import FileLock
from .log import logger

__all__ = [
    "COMBINED_UNSLOTH_NAME",
    "UNSLOTH_COMPILE_LOCATION",
    "UNSLOTH_COMPILE_USE_TEMP",
    "_get_compile_folder",
]

# Compiled cache location
global COMBINED_UNSLOTH_NAME
COMBINED_UNSLOTH_NAME = "unsloth_compiled_module"

global UNSLOTH_COMPILE_LOCATION
if 'UNSLOTH_COMPILE_LOCATION' not in globals():
    _loc = os.getenv("UNSLOTH_COMPILE_LOCATION", None)
    if _loc:
        UNSLOTH_COMPILE_LOCATION = _loc
    else:
        UNSLOTH_COMPILE_LOCATION = "unsloth_compiled_cache"

global UNSLOTH_COMPILE_USE_TEMP
UNSLOTH_COMPILE_USE_TEMP = False
UNSLOTH_COMPILE_LOCATION_FIXED = False

def _test_sentinel_write(location):
    try:
        lock_dir = Path(location) / ".locks"
        os.makedirs(lock_dir, exist_ok = True)
        lock = FileLock(str(lock_dir / "sentinel.lock"), timeout=5)
        with lock:
            sentinel_write = lock_dir / "sentinel_write"
            if not sentinel_write.exists():
                # try to write sentinel_write
                sentinel_write.write_text("sentinel test text")
    except Exception as e:
        raise e
    
def _get_compile_folder(use_tempfile = False):
    # known issue: if there is intermittent disk issues
    # multipe processes could end up with different compile locations
    # bad for locks, but a rare case
    global UNSLOTH_COMPILE_LOCATION
    global UNSLOTH_COMPILE_USE_TEMP
    global UNSLOTH_COMPILE_LOCATION_FIXED
    if UNSLOTH_COMPILE_LOCATION_FIXED:
        return UNSLOTH_COMPILE_LOCATION, UNSLOTH_COMPILE_USE_TEMP

    if UNSLOTH_COMPILE_USE_TEMP or use_tempfile:
        UNSLOTH_COMPILE_USE_TEMP = True
        leaf = os.path.basename(UNSLOTH_COMPILE_LOCATION)
        location = os.path.join(tempfile.gettempdir(), leaf)
        os.makedirs(location, exist_ok = True)
        # this can raise but since we don't have a fallback
        # it's a legit failure and we should raise
        _test_sentinel_write(location)
        UNSLOTH_COMPILE_LOCATION = location
        logger.info(
            f"Unsloth: We'll be using `{location}` for temporary Unsloth patches."
        )
        UNSLOTH_COMPILE_LOCATION_FIXED = True
    else:
        location = UNSLOTH_COMPILE_LOCATION
        try:
            # Try creating the directory
            os.makedirs(location, exist_ok = True)
            _test_sentinel_write(location)
            UNSLOTH_COMPILE_LOCATION_FIXED = True
            return location, UNSLOTH_COMPILE_USE_TEMP
        except Exception as e:
            logger.error(f"Unsloth: Failed to create directory `{UNSLOTH_COMPILE_LOCATION}` because {str(e)}")

            # Instead use a temporary location!
            location, UNSLOTH_COMPILE_USE_TEMP = _get_compile_folder(use_tempfile = True)
    return location, UNSLOTH_COMPILE_USE_TEMP
pass
