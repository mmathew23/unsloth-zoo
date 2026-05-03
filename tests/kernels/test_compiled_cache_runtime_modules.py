import sys
import types
import uuid

import unsloth_zoo.compiler as compiler


try:
    TRANSFORMERS_VERSION = compiler.importlib_version("transformers")
except Exception:
    TRANSFORMERS_VERSION = "0"


def test_create_new_function_can_load_isolated_runtime_module_instances(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    name = f"isolated_runtime_{uuid.uuid4().hex}"
    source = """
def generated_value():
    return marker()
"""

    first = compiler.create_new_function(
        name,
        source,
        "math",
        [],
        isolated_runtime_module = True,
    )
    second = compiler.create_new_function(
        name,
        source,
        "math",
        [],
        overwrite = False,
        isolated_runtime_module = True,
    )

    assert first is not second
    assert first.__file__ == second.__file__
    assert first.__name__ != second.__name__

    first.marker = lambda: "first"
    second.marker = lambda: "second"

    assert first.generated_value() == "first"
    assert second.generated_value() == "second"

    assert first.__name__ not in sys.modules
    assert second.__name__ not in sys.modules


def test_create_new_function_reloads_non_isolated_module_after_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    name = f"reload_runtime_{uuid.uuid4().hex}"

    try:
        first = compiler.create_new_function(
            name,
            """
def generated_value():
    return 1
""",
            "math",
            [],
        )
        second = compiler.create_new_function(
            name,
            """
def generated_value():
    return 2
""",
            "math",
            [],
            overwrite = True,
        )

        assert first.generated_value() == 1
        assert second.generated_value() == 2
    finally:
        sys.modules.pop(name, None)


def test_create_new_function_recompiles_cache_missing_backend_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    name = f"old_cache_metadata_{uuid.uuid4().hex}"
    cache_path = tmp_path / f"{name}.py"
    cache_path.write_text(
        '"""\n'
        '0\n'
        '0\n'
        f'{TRANSFORMERS_VERSION}\n'
        '0\n'
        '__UNSLOTH_VERSIONING__\n'
        '"""\n'
        "def generated_value():\n"
        "    return 1\n",
        encoding = "utf-8",
    )
    try:
        module = compiler.create_new_function(
            name,
            """
def generated_value():
    return 2
""",
            "math",
            [],
            overwrite = False,
        )
        assert module.generated_value() == 2
        assert compiler.UNSLOTH_COMPILE_BACKEND in cache_path.read_text(encoding = "utf-8")
    finally:
        sys.modules.pop(name, None)


def test_create_new_function_fails_fast_on_runtime_binding_error(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    name = f"runtime_binding_failure_{uuid.uuid4().hex}"

    fake_runtime_bindings = types.ModuleType("unsloth.kernels.runtime_bindings")

    def _raise_binding_error(module):
        raise ValueError("synthetic binding failure")

    fake_runtime_bindings.bind_kernel_runtime_globals = _raise_binding_error
    monkeypatch.setitem(
        sys.modules,
        "unsloth.kernels.runtime_bindings",
        fake_runtime_bindings,
    )

    try:
        try:
            compiler.create_new_function(
                name,
                """
def generated_value():
    return 1
""",
                "math",
                [],
                bind_kernel_runtime_globals = True,
            )
        except RuntimeError as exc:
            assert "failed to bind kernel runtime globals" in str(exc)
            assert isinstance(exc.__cause__, ValueError)
        else:
            raise AssertionError("runtime binding failure was swallowed")
    finally:
        sys.modules.pop(name, None)
