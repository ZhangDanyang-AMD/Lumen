"""SFT dataset for ChatML-format instruction-response pairs.

Reads JSONL with ``{"messages": [...], ...}`` records, applies the tokenizer's
chat template, and produces input_ids + labels with loss masked on non-assistant
tokens (system + user turns).
"""

import json
import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

__all__ = ["SFTDataset"]

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """SFT dataset with answer-only loss masking.

    Args:
        data_path: Path to a ``.jsonl`` file with ``messages`` field.
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        max_seq_length: Truncate sequences longer than this.
    """

    def __init__(
        self,
        data_path: Optional[str],
        tokenizer,
        max_seq_length: int = 8192,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self._samples: List[Dict[str, torch.Tensor]] = []

        if data_path is None:
            return

        with open(data_path, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]

        skipped = 0
        for record in raw:
            messages = record.get("messages", [])
            if not messages:
                skipped += 1
                continue

            sample = self._process(messages)
            if sample is not None:
                self._samples.append(sample)
            else:
                skipped += 1

        logger.info(
            "SFTDataset: %d samples from %s (%d skipped)",
            len(self._samples), data_path, skipped,
        )

    def _process(self, messages: List[dict]) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize messages with chat template and create loss mask."""
        # Tokenize full conversation
        full_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )

        if len(full_ids) < 2:
            return None

        # Truncate
        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[:self.max_seq_length]

        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        labels = torch.tensor(full_ids[1:], dtype=torch.long)

        # Build loss mask: only compute loss on assistant tokens.
        # Tokenize everything except assistant turns to find boundaries.
        non_assistant_ids = self.tokenizer.apply_chat_template(
            [m for m in messages if m["role"] != "assistant"],
            tokenize=True, add_generation_prompt=True,
        )
        non_assistant_len = len(non_assistant_ids)

        # Simple heuristic: tokenize up to each assistant turn boundary
        loss_mask = torch.zeros(len(labels), dtype=torch.float32)
        prefix_len = 0
        for i, msg in enumerate(messages):
            # Tokenize messages up to and including this one
            partial = self.tokenizer.apply_chat_template(
                messages[:i+1], tokenize=True, add_generation_prompt=False,
            )
            end_pos = min(len(partial) - 1, len(labels))

            if msg["role"] == "assistant":
                # Mark assistant tokens for loss
                start_pos = prefix_len
                loss_mask[start_pos:end_pos] = 1.0

            prefix_len = end_pos

        return {"input_ids": input_ids, "labels": labels, "loss_mask": loss_mask}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._samples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad batch to max length within batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    loss_mask = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        loss_mask[i, :seq_len] = b["loss_mask"]

    return {"input_ids": input_ids, "labels": labels, "loss_mask": loss_mask}
