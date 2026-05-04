import os
import importlib
import inspect
import re
import sys
import unittest
import warnings
from contextlib import contextmanager
from unittest.mock import patch

import unsloth_zoo.compile_policy as compile_policy
from unsloth_zoo.temporary_patches import common


@contextmanager
def _expect_single_warning(testcase, category, pattern):
    compiled = re.compile(pattern)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        yield

    matches = [
        warning
        for warning in caught
        if issubclass(warning.category, category) and compiled.search(str(warning.message))
    ]
    unexpected = [
        warning
        for warning in caught
        if warning not in matches
    ]
    testcase.assertEqual(
        len(matches),
        1,
        f"Expected exactly one {category.__name__} matching {pattern!r}; "
        f"captured {[str(w.message) for w in caught]}",
    )
    testcase.assertEqual(
        unexpected,
        [],
        f"Unexpected warnings captured: {[str(w.message) for w in unexpected]}",
    )


class CompileBackendCommonTests(unittest.TestCase):
    def test_detect_compile_backend_prefers_explicit_env(self):
        with patch.dict(
            os.environ,
            {
                "UNSLOTH_TORCH_COMPILE_BACKEND": "aot_eager",
                "TORCHDYNAMO_DISABLE": "1",
            },
        ):
            self.assertEqual(compile_policy._detect_compile_backend(), "aot_eager")
            self.assertNotIn("TORCHDYNAMO_DISABLE", os.environ)

    def test_detect_compile_backend_normalizes_explicit_env(self):
        with patch.dict(os.environ, {"UNSLOTH_TORCH_COMPILE_BACKEND": " AOT-EAGER "}):
            self.assertEqual(compile_policy._detect_compile_backend(), "aot_eager")

        with patch.dict(os.environ, {"UNSLOTH_TORCH_COMPILE_BACKEND": "INDUCTOR"}):
            self.assertEqual(compile_policy._detect_compile_backend(), "inductor")

        with patch.dict(os.environ, {"UNSLOTH_TORCH_COMPILE_BACKEND": "dynamo-disabled"}):
            self.assertEqual(compile_policy._detect_compile_backend(), "dynamo_disable")

        with patch.dict(
            os.environ,
            {
                "UNSLOTH_TORCH_COMPILE_BACKEND": "dynamo_disable",
                "TORCHDYNAMO_DISABLE": "0",
            },
        ):
            self.assertEqual(compile_policy._detect_compile_backend(), "dynamo_disable")
            self.assertEqual(os.environ["TORCHDYNAMO_DISABLE"], "1")

    def test_detect_compile_backend_uses_dynamo_disable_without_triton(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(compile_policy, "_is_triton_importable", return_value = False):
                self.assertEqual(compile_policy._detect_compile_backend(), "dynamo_disable")
                self.assertEqual(os.environ["TORCHDYNAMO_DISABLE"], "1")

    def test_detect_compile_backend_uses_dynamo_disable_when_triton_import_fails(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("importlib.import_module", side_effect = OSError("broken triton")):
                self.assertEqual(compile_policy._detect_compile_backend(), "dynamo_disable")
                self.assertEqual(os.environ["TORCHDYNAMO_DISABLE"], "1")

    def test_kernel_backend_triton_does_not_disable_inductor_compile(self):
        with patch.dict(
            os.environ,
            {
                "UNSLOTH_KERNEL_BACKEND": "triton",
                "TORCHDYNAMO_DISABLE": "1",
            },
            clear = True,
        ):
            with patch.object(compile_policy, "_is_triton_importable", return_value = True):
                self.assertEqual(compile_policy._detect_compile_backend(), "inductor")
                self.assertNotIn("TORCHDYNAMO_DISABLE", os.environ)

    def test_make_torch_compile_keeps_inductor_call_shape_default(self):
        captured = {}

        def fake_compile(fn = None, **kwargs):
            captured.update(kwargs)
            return fn if fn is not None else (lambda f: f)

        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "inductor"):
            with patch.object(common.torch, "compile", fake_compile):
                decorator = common._make_torch_compile({"trace.enabled": False})
                decorator(lambda x: x)

        self.assertNotIn("backend", captured)
        self.assertEqual(captured["options"], {"trace.enabled": False})

    def test_make_torch_compile_routes_non_inductor_explicitly(self):
        captured = {}

        def fake_compile(fn = None, **kwargs):
            captured.update(kwargs)
            return fn if fn is not None else (lambda f: f)

        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "aot_eager"):
            with patch.object(common.torch, "compile", fake_compile):
                decorator = common._make_torch_compile({"trace.enabled": False})
                decorator(lambda x: x, mode = "max-autotune")

        self.assertEqual(captured["backend"], "aot_eager")
        self.assertNotIn("options", captured)
        self.assertNotIn("mode", captured)

    def test_make_torch_compile_dynamo_disable_does_not_call_torch_compile(self):
        called = {"compile": False, "noop": False}

        def fake_compile(*args, **kwargs):
            called["compile"] = True
            raise AssertionError("dynamo_disable must not call torch.compile")

        def fake_noop(*args, **kwargs):
            called["noop"] = True
            return args[0] if args and callable(args[0]) else (lambda f: f)

        def fn(x):
            return x

        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "dynamo_disable"):
            with patch.object(common.torch, "compile", fake_compile):
                with patch.object(common, "noop", fake_noop):
                    decorator = common._make_torch_compile({"trace.enabled": False})
                    compiled = decorator(fn, mode = "max-autotune")

        self.assertIs(compiled, fn)
        self.assertTrue(called["noop"])
        self.assertFalse(called["compile"])

    def test_make_torch_compile_non_inductor_decorator_falls_back_to_eager(self):
        def boom_compile(*args, **kwargs):
            raise OSError("broken compile backend")

        def fn(x):
            return x

        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "aot_eager"):
            with patch.object(common.torch, "compile", boom_compile):
                decorator = common._make_torch_compile({"trace.enabled": False})
                with _expect_single_warning(
                    self,
                    UserWarning,
                    r"torch\.compile backend 'aot_eager' failed .* falling back to eager",
                ):
                    compiled = decorator(dynamic = True)(fn)

        self.assertIs(compiled, fn)

    def test_flex_attention_reports_disabled_on_non_inductor_backend(self):
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
            with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "aot_eager"):
                with patch("torch.cuda.device_count", return_value = 1):
                    with patch(
                        "torch.cuda.memory.mem_get_info",
                        return_value = (0, 16 * 1024 * 1024 * 1024),
                    ):
                        with _expect_single_warning(
                            self,
                            UserWarning,
                            r"flex_attention disabled: requires inductor .* Falling back to SDPA",
                        ):
                            flex_utils = importlib.import_module("unsloth_zoo.flex_attention.utils")
            self.assertFalse(flex_utils.HAS_FLEX_ATTENTION)
            self.assertIsNone(flex_utils.flex_attention)
        finally:
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

    def test_generated_cache_source_is_backend_neutral_on_inductor(self):
        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "inductor"):
            decorator = compile_policy.get_torch_compile_decorator_source(
                fullgraph = True,
                dynamic = True,
            )
            compile_import = compile_policy.get_torch_compile_import_source()

        self.assertEqual(
            decorator,
            "@_unsloth_torch_compile(fullgraph = True, dynamic = True)",
        )
        self.assertIn("torch_compile as _unsloth_torch_compile", compile_import)

    def test_generated_cache_source_is_backend_neutral_on_non_inductor(self):
        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "aot_eager"):
            decorator = compile_policy.get_torch_compile_decorator_source(
                fullgraph = False,
                dynamic = True,
            )
            compile_import = compile_policy.get_torch_compile_import_source()

        self.assertEqual(
            decorator,
            "@_unsloth_torch_compile(fullgraph = False, dynamic = True)",
        )
        self.assertIn("torch_compile as _unsloth_torch_compile", compile_import)

    def test_generated_cache_policy_can_target_explicit_backend_without_changing_source(self):
        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "aot_eager"):
            decorator = compile_policy.get_torch_compile_decorator_source(
                fullgraph = True,
                dynamic = None,
                backend = " INDUCTOR ",
                wrapper_name = "_compile_for_decode",
            )

        self.assertEqual(
            decorator,
            "@_compile_for_decode(fullgraph = True, dynamic = None)",
        )

    def test_generated_cache_policy_accepts_explicit_wrapper_backend(self):
        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "inductor"):
            decorator = compile_policy.get_torch_compile_decorator_source(
                backend = "aot-eager",
                wrapper_name = "_compile_for_decode",
            )

        self.assertEqual(
            decorator,
            "@_compile_for_decode(fullgraph = True, dynamic = True)",
        )

    def test_generated_cache_policy_normalizes_global_backend(self):
        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", " INDUCTOR "):
            self.assertFalse(compile_policy.torch_compile_uses_direct_source())
            decorator = compile_policy.get_torch_compile_decorator_source(
                fullgraph = True,
                dynamic = True,
            )

        self.assertEqual(
            decorator,
            "@_unsloth_torch_compile(fullgraph = True, dynamic = True)",
        )

    def test_generated_cache_policy_source_matches_across_compile_backends(self):
        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "inductor"):
            inductor_source = (
                compile_policy.get_torch_compile_import_source()
                + compile_policy.get_torch_compile_decorator_source(
                    fullgraph = True,
                    dynamic = True,
                )
            )

        with patch.object(compile_policy, "UNSLOTH_COMPILE_BACKEND", "aot_eager"):
            aot_eager_source = (
                compile_policy.get_torch_compile_import_source()
                + compile_policy.get_torch_compile_decorator_source(
                    fullgraph = True,
                    dynamic = True,
                )
            )

        self.assertEqual(inductor_source, aot_eager_source)

    def test_generated_cache_policy_does_not_mutate_temporary_patch_registry(self):
        before = tuple(common.TEMPORARY_PATCHES)

        compile_policy.get_torch_compile_decorator_source()
        compile_policy.get_torch_compile_import_source()
        compile_policy.torch_compile_uses_direct_source()

        self.assertEqual(tuple(common.TEMPORARY_PATCHES), before)

    def test_loss_patch_compile_helper_is_not_shadowed_by_bool_parameter(self):
        import unsloth_zoo.loss_utils as loss_utils

        source = inspect.getsource(loss_utils.patch_loss_functions)

        self.assertIn("torch_compile = True", source)
        self.assertIn("_module_torch_compile(", source)
        self.assertNotIn("UnslothForCausalLMLoss = torch_compile(", source)

    def test_patch_function_uses_backend_aware_compile_helper(self):
        from unsloth_zoo.temporary_patches import utils

        class Target:
            @staticmethod
            def forward(x):
                return x

        def replacement(x):
            return x

        calls = []

        def fake_torch_compile(*args, **kwargs):
            raise AssertionError("patch_function must not call raw torch.compile")

        def fake_module_torch_compile(fn, **kwargs):
            calls.append(kwargs)

            def compiled(*args, **inner_kwargs):
                return fn(*args, **inner_kwargs)

            return compiled

        with patch.object(utils.torch, "compile", fake_torch_compile):
            with patch.object(utils, "torch_compile", fake_module_torch_compile):
                patched = utils.patch_function(
                    Target,
                    "forward",
                    replacement,
                    force = True,
                    fullgraph = True,
                    dynamic = False,
                )

        self.assertTrue(patched)
        self.assertEqual(calls, [{"fullgraph": True, "dynamic": False}])


if __name__ == "__main__":
    unittest.main()
