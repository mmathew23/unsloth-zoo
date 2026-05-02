import textwrap
import unittest
from pathlib import Path

from unsloth_zoo.compiler import patch_gpt_oss_dict_attention_mask


ROOT = Path(__file__).resolve().parents[2]


class TestGptOssAttentionMaskCompilePatch(unittest.TestCase):
    def test_dict_mask_patch_slices_to_kv_length(self):
        source = textwrap.dedent(
            """
            def eager_attention_forward(module, query, key, value, attention_mask, scaling):
                key_states = repeat_kv(key, module.num_key_value_groups)
                attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                return attn_weights
            """
        )

        patched = patch_gpt_oss_dict_attention_mask(source, model_type = "gpt_oss")

        self.assertIn("if isinstance(attention_mask, dict):", patched)
        self.assertIn(
            "attention_mask = attention_mask.get(getattr(module, 'layer_type', None), None)",
            patched,
        )
        self.assertIn(
            "attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]",
            patched,
        )
        self.assertIn("attn_weights = attn_weights + attention_mask", patched)

    def test_patch_requires_kv_length_variable(self):
        source = textwrap.dedent(
            """
            def eager_attention_forward(module, query, key, value, attention_mask, scaling):
                attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                return attn_weights
            """
        )

        self.assertEqual(
            patch_gpt_oss_dict_attention_mask(source, model_type = "gpt_oss"),
            source,
        )

    def test_patch_is_scoped_to_gpt_oss(self):
        source = textwrap.dedent(
            """
            def eager_attention_forward(module, query, key, value, attention_mask, scaling):
                key_states = repeat_kv(key, module.num_key_value_groups)
                attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                return attn_weights
            """
        )

        self.assertEqual(
            patch_gpt_oss_dict_attention_mask(source, model_type = "biogpt"),
            source,
        )


class TestGptOssTemporaryPatchSource(unittest.TestCase):
    def test_model_forward_selects_attention_mask_by_layer_type(self):
        source = (ROOT / "unsloth_zoo/temporary_patches/gpt_oss.py").read_text()

        self.assertNotIn('getattr(decoder_layer, "attention_type", None)', source)
        self.assertIn('getattr(getattr(decoder_layer, "self_attn", None), "layer_type", None)', source)
        self.assertIn("self.config.layer_types[i]", source)

    def test_gpt_oss_flex_attention_patch_is_disabled_only_off_inductor_by_default(self):
        source = (ROOT / "unsloth_zoo/temporary_patches/gpt_oss.py").read_text()

        self.assertIn("GPT-OSS Flex Attention patch is disabled", source)
        self.assertIn('return UNSLOTH_COMPILE_BACKEND == "inductor"', source)
        self.assertIn("_print_gpt_oss_flex_attention_disabled_once()", source)
        self.assertIn("self.training and _gpt_oss_flex_attention_patch_enabled()", source)


if __name__ == "__main__":
    unittest.main()
