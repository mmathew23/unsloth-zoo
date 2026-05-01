import os
import unittest
from unittest.mock import patch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.temporary_patches import common


class CompileBackendCommonTests(unittest.TestCase):
    def test_detect_compile_backend_prefers_explicit_env(self):
        with patch.dict(os.environ, {"UNSLOTH_TORCH_COMPILE_BACKEND": "aot_eager"}):
            self.assertEqual(common._detect_compile_backend(), "aot_eager")

    def test_detect_compile_backend_uses_aot_eager_without_triton(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("importlib.util.find_spec", return_value = None):
                self.assertEqual(common._detect_compile_backend(), "aot_eager")

    def test_make_torch_compile_keeps_inductor_call_shape_default(self):
        captured = {}

        def fake_compile(fn = None, **kwargs):
            captured.update(kwargs)
            return fn if fn is not None else (lambda f: f)

        previous_backend = common.UNSLOTH_COMPILE_BACKEND
        try:
            common.UNSLOTH_COMPILE_BACKEND = "inductor"
            with patch.object(common.torch, "compile", fake_compile):
                decorator = common._make_torch_compile({"trace.enabled": False})
                decorator(lambda x: x)
        finally:
            common.UNSLOTH_COMPILE_BACKEND = previous_backend

        self.assertNotIn("backend", captured)
        self.assertEqual(captured["options"], {"trace.enabled": False})

    def test_make_torch_compile_routes_non_inductor_explicitly(self):
        captured = {}

        def fake_compile(fn = None, **kwargs):
            captured.update(kwargs)
            return fn if fn is not None else (lambda f: f)

        previous_backend = common.UNSLOTH_COMPILE_BACKEND
        try:
            common.UNSLOTH_COMPILE_BACKEND = "aot_eager"
            with patch.object(common.torch, "compile", fake_compile):
                decorator = common._make_torch_compile({"trace.enabled": False})
                decorator(lambda x: x, mode = "max-autotune")
        finally:
            common.UNSLOTH_COMPILE_BACKEND = previous_backend

        self.assertEqual(captured["backend"], "aot_eager")
        self.assertNotIn("options", captured)
        self.assertNotIn("mode", captured)


if __name__ == "__main__":
    unittest.main()
