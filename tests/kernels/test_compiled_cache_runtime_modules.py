import uuid

import unsloth_zoo.compiler as compiler


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
