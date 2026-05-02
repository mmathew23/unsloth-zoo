import os
import importlib
import sys
import unittest
from unittest.mock import patch

from unsloth_zoo.temporary_patches import common


class CompileBackendCommonTests(unittest.TestCase):
    def test_detect_compile_backend_prefers_explicit_env(self):
        with patch.dict(os.environ, {"UNSLOTH_TORCH_COMPILE_BACKEND": "aot_eager"}):
            self.assertEqual(common._detect_compile_backend(), "aot_eager")

    def test_detect_compile_backend_uses_aot_eager_without_triton(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(common, "_is_triton_importable", return_value = False):
                self.assertEqual(common._detect_compile_backend(), "aot_eager")

    def test_detect_compile_backend_uses_aot_eager_when_triton_import_fails(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("importlib.import_module", side_effect = OSError("broken triton")):
                self.assertEqual(common._detect_compile_backend(), "aot_eager")

    def test_kernel_backend_triton_does_not_disable_inductor_compile(self):
        with patch.dict(
            os.environ,
            {
                "UNSLOTH_KERNEL_BACKEND": "triton",
            },
            clear = True,
        ):
            with patch.object(common, "_is_triton_importable", return_value = True):
                self.assertEqual(common._detect_compile_backend(), "inductor")

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

    def test_make_torch_compile_non_inductor_decorator_falls_back_to_eager(self):
        def boom_compile(*args, **kwargs):
            raise OSError("broken compile backend")

        def fn(x):
            return x

        previous_backend = common.UNSLOTH_COMPILE_BACKEND
        try:
            common.UNSLOTH_COMPILE_BACKEND = "aot_eager"
            with patch.object(common.torch, "compile", boom_compile):
                decorator = common._make_torch_compile({"trace.enabled": False})
                compiled = decorator(dynamic = True)(fn)
        finally:
            common.UNSLOTH_COMPILE_BACKEND = previous_backend

        self.assertIs(compiled, fn)

    def test_flex_attention_reports_disabled_on_non_inductor_backend(self):
        previous_backend = common.UNSLOTH_COMPILE_BACKEND
        module_names = [
            "unsloth_zoo.flex_attention",
            "unsloth_zoo.flex_attention.utils",
        ]
        previous_modules = {
            name: sys.modules.pop(name)
            for name in module_names
            if name in sys.modules
        }
        try:
            common.UNSLOTH_COMPILE_BACKEND = "aot_eager"
            with patch("torch.cuda.device_count", return_value = 1):
                with patch(
                    "torch.cuda.memory.mem_get_info",
                    return_value = (0, 16 * 1024 * 1024 * 1024),
                ):
                    flex_utils = importlib.import_module("unsloth_zoo.flex_attention.utils")
            self.assertFalse(flex_utils.HAS_FLEX_ATTENTION)
            self.assertIsNone(flex_utils.flex_attention)
        finally:
            common.UNSLOTH_COMPILE_BACKEND = previous_backend
            for name in module_names:
                sys.modules.pop(name, None)
            sys.modules.update(previous_modules)

    def test_fused_lm_head_patch_allows_missing_triton_on_aot_eager(self):
        from unsloth_zoo import compiler

        with patch.multiple(
            compiler,
            OLD_CUDA_ARCH_VERSION = False,
            OLD_TORCH_VERSION = False,
            OLD_TRITON_VERSION = True,
            UNSLOTH_COMPILE_BACKEND = "aot_eager",
        ):
            self.assertFalse(compiler.should_skip_fused_lm_head_patch())

    def test_fused_lm_head_patch_skips_missing_triton_on_inductor(self):
        from unsloth_zoo import compiler

        with patch.multiple(
            compiler,
            OLD_CUDA_ARCH_VERSION = False,
            OLD_TORCH_VERSION = False,
            OLD_TRITON_VERSION = True,
            UNSLOTH_COMPILE_BACKEND = "inductor",
        ):
            self.assertTrue(compiler.should_skip_fused_lm_head_patch())


if __name__ == "__main__":
    unittest.main()
