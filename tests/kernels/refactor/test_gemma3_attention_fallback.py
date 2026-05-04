import unittest


class Gemma3AttentionFallbackTests(unittest.TestCase):
    def test_non_tensor_flex_mask_is_removed_for_sdpa_fallback(self):
        from unsloth_zoo.temporary_patches import gemma

        class FakeBlockMask:
            pass

        self.assertIsNone(gemma._gemma3_sdpa_attention_mask(FakeBlockMask()))

    def test_tensor_mask_is_preserved_for_sdpa_fallback(self):
        import torch
        from unsloth_zoo.temporary_patches import gemma

        mask = torch.ones((1, 1, 2, 2), dtype = torch.bool)
        self.assertIs(gemma._gemma3_sdpa_attention_mask(mask), mask)


if __name__ == "__main__":
    unittest.main()
