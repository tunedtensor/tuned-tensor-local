from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_contract import assert_certified_model_config


class ModelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "intermediate_size": 6144,
                "vocab_size": 248320,
            },
        }

    def test_accepts_the_certified_2b_architecture(self) -> None:
        assert_certified_model_config(self.config)
        assert_certified_model_config(self.config["text_config"])

    def test_rejects_a_larger_same_family_snapshot(self) -> None:
        larger = {
            **self.config,
            "text_config": {
                **self.config["text_config"],
                "hidden_size": 2560,
            },
        }
        with self.assertRaisesRegex(ValueError, "Qwen/Qwen3.5-2B"):
            assert_certified_model_config(larger)


if __name__ == "__main__":
    unittest.main()
