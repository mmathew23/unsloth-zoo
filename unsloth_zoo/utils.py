# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "Version",
    "_get_dtype",
    "is_main_process",
    "is_distributed",
    "distributed_function",
    "torch_distributed_get_rank",
    "get_compile_folder",
    "get_lock",
]

from packaging.version import Version as TrueVersion
import torch
import os
import time
import contextlib
import re
import pathlib
from typing import Optional
from filelock import FileLock
import zlib
from .globals import _get_compile_folder

def Version(version):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        version = str(version)
        try:
            return TrueVersion(version)
        except Exception as e:
            version = re.match(r"[0-9\.]{1,}", version)
            if version is None:
                raise Exception(str(e))
            version = version.group(0).rstrip(".")
            return TrueVersion(version)
    except:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Unsloth: Could not get version for `{version}`\n"\
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )
    pass
pass


__DTYPE_MAP = {
    "float32": torch.float32,
    torch.float32: torch.float32,
    "float16": torch.float16,
    torch.float16: torch.float16,
    "bfloat16": torch.bfloat16,
    torch.bfloat16: torch.bfloat16,
}
def _get_dtype(dtype):
    try:
        return __DTYPE_MAP[dtype]
    except:
        if type(dtype) is str:
            dtype = dtype.lower()
            return getattr(torch, dtype, None)
        elif isinstance(dtype, torch.dtype):
            return dtype
    return None
pass


import functools
torch_distributed_is_initialized = torch.distributed.is_initialized
torch_distributed_is_torchelastic_launched = torch.distributed.is_torchelastic_launched
torch_distributed_get_rank = torch.distributed.get_rank

def is_main_process():
    if torch_distributed_is_initialized():
        # torch.distributed.init_process_group was run, so get_rank works
        return torch_distributed_get_rank() == 0
    elif torch_distributed_is_torchelastic_launched():
        # accelerate launch for example calls init_process_group later
        return os.environ.get("RANK", "0") == "0"
    return True
pass

def is_distributed():
    return torch_distributed_is_initialized() or torch_distributed_is_torchelastic_launched()
pass

def distributed_function(n = 1, function = None, *args, **kwargs):
    if is_distributed():
        if is_main_process():
            object_list = function(*args, **kwargs)
            if n == 1: object_list = [object_list]
        else:
            object_list = [None for _ in range(n)]
        # broadcast_object_list auto blocks so no need for barrier
        if not torch_distributed_is_initialized():
            # But check if the function even works!
            # This happens when torch_distributed_is_torchelastic_launched()==True but
            # torch_distributed_is_initialized()==False
            # Trick is to just add a 0.01+0.01*RANK second sleep and print with flush
            time.sleep(0.01 + 0.01*int(os.environ.get("RANK", "0")))
            with contextlib.redirect_stdout(None):
                print("", flush = True)
            object_list = function(*args, **kwargs)
            if n == 1: object_list = [object_list]
        else:
            torch.distributed.broadcast_object_list(object_list, src = 0)
        if n == 1:
            result = object_list[0]
        else:
            result = object_list
    else:
        result = function(*args, **kwargs)
    return result
pass

def _canon_key(p: str) -> str:
    s = os.path.abspath(p)
    if os.name == "nt":
        s = os.path.normcase(s)
    return os.path.normpath(s)

def _slug(name: str, maxlen: int = 100) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return (name or "_")[:maxlen]

def _lock_name_for(target: str) -> str:
    canon = _canon_key(target)
    base  = _slug(pathlib.Path(canon).name)
    h8    = f"{zlib.crc32(canon.encode('utf-8')) & 0xffffffff:08x}"
    return f"{base}.{h8}.lock"

def _lock_path_for(target: str) -> str:
    """ str needs to be a valid file path """
    base_dir = _get_compile_folder()[0]
    locks_dir = pathlib.Path(base_dir) / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_name = _lock_name_for(target)
    return str(locks_dir / lock_name)

def get_lock(target: str, timeout: Optional[int] = None) -> FileLock:
    """
    Get a lock for a target file.
    target: str, the path to the file to lock
    timeout: int, the timeout in seconds for the lock
    If timeout is not provided, it will use the value of
    the environment variable UNSLOTH_LOCK_TIMEOUT, otherwise 10 seconds.

    Returns:
        FileLock, the lock for the target file
    """
    lock_path = _lock_path_for(target)
    if timeout is None:
        timeout = int(os.environ.get("UNSLOTH_LOCK_TIMEOUT", "10"))
    return FileLock(lock_path, timeout=timeout)

def get_compile_folder(use_tempfile = False, distributed = True):
    if distributed:
        location, UNSLOTH_COMPILE_USE_TEMP = distributed_function(2, _get_compile_folder, use_tempfile)
    else:
        location, UNSLOTH_COMPILE_USE_TEMP = _get_compile_folder(use_tempfile)

    return location, UNSLOTH_COMPILE_USE_TEMP
pass

@contextlib.contextmanager
def locked_path(path):
    lock = get_lock(path)
    with lock:
        yield

@contextlib.contextmanager
def open_locked(path,
                mode="r",
                buffering=-1,
                encoding=None,
                errors=None,
                newline=None):
    kw = {}
    if "b" not in mode:
        kw["encoding"] = encoding or "utf-8"
        if errors is not None:
            kw["errors"] = errors
        if newline is not None:
            kw["newline"] = newline

    with locked_path(path):
        f = open(path, mode, buffering=buffering, **kw)
        try:
            yield f
        finally:
            f.close()

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
