from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sft_data import IGNORE_INDEX, build_assistant_only_example


class FakeQwenTokenizer:
    ROLE_TOKENS = {
        "system": 101,
        "user": 102,
        "assistant": 103,
        "tool": 104,
    }
    END_TOKEN = 105
    GENERATION_TOKEN = 106

    @staticmethod
    def _content_tokens(content: Any) -> list[int]:
        return [200 + ord(character) for character in str(content)]

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        **_: Any,
    ) -> list[int]:
        if not tokenize or return_dict:
            raise AssertionError("tests expect direct token-id rendering")
        tokens: list[int] = []
        for message in messages:
            tokens.append(self.ROLE_TOKENS[message["role"]])
            if message["role"] == "assistant":
                tokens.append(self.GENERATION_TOKEN)
            tokens.extend(self._content_tokens(message.get("content", "")))
            tokens.append(self.END_TOKEN)
        if add_generation_prompt:
            tokens.extend([self.ROLE_TOKENS["assistant"], self.GENERATION_TOKEN])
        return tokens


class BrokenTemplateTokenizer(FakeQwenTokenizer):
    def apply_chat_template(self, *args: Any, **kwargs: Any) -> list[int]:
        raise RuntimeError("template mismatch")


class AssistantOnlyExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeQwenTokenizer()
        self.messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

    def test_masks_every_prompt_token_and_labels_the_complete_assistant_turn(self) -> None:
        example = build_assistant_only_example(
            self.tokenizer,
            self.messages,
            max_length=128,
        )
        prompt = self.tokenizer.apply_chat_template(
            self.messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        completion = full[len(prompt) :]

        self.assertEqual(example["input_ids"], full)
        self.assertEqual(example["labels"][: len(prompt)], [IGNORE_INDEX] * len(prompt))
        self.assertEqual(example["labels"][len(prompt) :], completion)
        self.assertEqual(example["attention_mask"], [1] * len(full))

    def test_truncates_old_prompt_tokens_without_dropping_any_answer_tokens(self) -> None:
        long_messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "0123456789" * 20},
            {"role": "assistant", "content": "answer"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            long_messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full = self.tokenizer.apply_chat_template(
            long_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        completion = full[len(prompt) :]
        max_length = len(completion) + 12

        example = build_assistant_only_example(
            self.tokenizer,
            long_messages,
            max_length=max_length,
        )

        self.assertEqual(len(example["input_ids"]), max_length)
        self.assertEqual(example["input_ids"][-len(completion) :], completion)
        self.assertEqual(example["labels"][-len(completion) :], completion)
        self.assertEqual(example["labels"][:-len(completion)], [IGNORE_INDEX] * 12)
        self.assertEqual(example["input_ids"][:12], prompt[-12:])

    def test_rejects_an_answer_that_cannot_fit_without_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "answer is not truncated"):
            build_assistant_only_example(
                self.tokenizer,
                self.messages,
                max_length=5,
            )

    def test_requires_the_last_turn_to_be_a_nonempty_assistant_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "final message must have role=assistant"):
            build_assistant_only_example(
                self.tokenizer,
                self.messages[:-1],
                max_length=128,
            )

    def test_rejects_structured_message_content(self) -> None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": "hi"},
        ]
        with self.assertRaisesRegex(ValueError, "text-only"):
            build_assistant_only_example(
                self.tokenizer,
                messages,
                max_length=128,
            )

    def test_surfaces_chat_template_failures(self) -> None:
        with self.assertRaisesRegex(ValueError, "template mismatch"):
            build_assistant_only_example(
                BrokenTemplateTokenizer(),
                self.messages,
                max_length=128,
            )


if __name__ == "__main__":
    unittest.main()
