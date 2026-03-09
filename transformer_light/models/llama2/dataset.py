###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Framework-agnostic SFT dataset for LLaMA2.

Shared between the Megatron and FSDP training backends.  Has no dependency
on Megatron, HuggingFace Transformers, or any training framework — only
PyTorch and the Python standard library.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["LLaMA2SFTDataset"]

logger = logging.getLogger(__name__)


class LLaMA2SFTDataset(Dataset):
    """SFT dataset that loads jsonl data and packs sequences.

    Each jsonl line should have the format::

        {"input": "<prompt text>", "output": "<completion text>"}

    or::

        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Sequences are tokenized, packed to ``seq_length``, and loss is masked
    so that only the completion (output/assistant) tokens contribute.

    Args:
        num_samples: Number of samples the dataset should report via ``__len__``.
        data_path: Path to a ``.jsonl`` or ``.json`` file.
        seq_length: Maximum sequence length (tokens will be packed to this).
        tokenizer: Any object with an ``encode(text) -> list[int]`` method
            and an ``eos_token_id`` attribute, **or** a Megatron tokenizer
            with ``tokenize()`` / ``eod``.
        is_hf_tokenizer: ``True`` if the tokenizer has HuggingFace-style API.
    """

    LLAMA2_CHAT_TEMPLATE = "[INST] {input} [/INST] {output}"

    def __init__(
        self,
        num_samples: int,
        data_path: Optional[str],
        seq_length: int,
        tokenizer,
        is_hf_tokenizer: bool = False,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer
        self.indexed_dataset: List[Dict[str, list]] = []
        self._raw_idx = 0

        if data_path is None:
            self._raw_samples: list = []
            return

        if not os.path.isfile(data_path):
            alt = data_path.rsplit(".", 1)[0] + ".npy"
            if data_path.endswith((".jsonl", ".json")) and os.path.isfile(alt):
                data_path = alt
                logger.info("Using %s (requested path not found)", data_path)

        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                self._raw_samples = [json.loads(line) for line in f if line.strip()]
        elif data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                self._raw_samples = json.load(f)
        elif data_path.endswith(".npy"):
            arr = np.load(data_path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                ids_arr = arr.astype(np.int64)
                mask_arr = np.ones_like(ids_arr, dtype=np.int64)
                self._raw_samples = [
                    {"input_ids": ids_arr[i].tolist(), "loss_mask": mask_arr[i].tolist()}
                    for i in range(len(ids_arr))
                ]
            else:
                raise ValueError(f".npy must be 2D array (N, seq_len), got shape {getattr(arr, 'shape', 'unknown')}")
        elif data_path.endswith(".npz"):
            data = np.load(data_path, allow_pickle=True)
            ids_arr = data["input_ids"].astype(np.int64)
            mask_arr = data.get("loss_mask")
            if mask_arr is None:
                mask_arr = np.ones_like(ids_arr, dtype=np.int64)
            else:
                mask_arr = mask_arr.astype(np.int64)
            self._raw_samples = [
                {"input_ids": ids_arr[i].tolist(), "loss_mask": mask_arr[i].tolist()}
                for i in range(len(ids_arr))
            ]
        else:
            raise ValueError(f"Unsupported data format: {data_path} (use .jsonl, .json, .npy, or .npz)")

        logger.info("Loaded %d samples from %s", len(self._raw_samples), data_path)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        while idx >= len(self.indexed_dataset):
            packed = self._pack_next()
            if packed is None:
                break
            self.indexed_dataset.append(packed)

        idx = idx % max(len(self.indexed_dataset), 1)
        sample = self.indexed_dataset[idx]
        return {k: torch.LongTensor(v) for k, v in sample.items()}

    # -- internal helpers --

    def _tokenize(self, text: str) -> List[int]:
        if self.is_hf_tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.tokenize(text)

    def _get_eos_id(self) -> int:
        if self.is_hf_tokenizer:
            return self.tokenizer.eos_token_id
        return self.tokenizer.eod

    def _process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Tokenize one raw sample and compute the answer-only loss mask."""
        if "input_ids" in sample:
            ids = sample["input_ids"]
            mask = sample.get("loss_mask", [1] * len(ids))
            if len(ids) > self.seq_length:
                ids = ids[: self.seq_length]
                mask = mask[: self.seq_length]
            return {"input_ids": ids, "loss_mask": mask, "token_count": len(ids)}
        if "messages" in sample:
            messages = sample["messages"]
            if len(messages) < 2:
                return None
            prompt_parts, completion_parts = [], []
            for msg in messages:
                if msg["role"] in ("user", "system"):
                    prompt_parts.append(msg["content"])
                elif msg["role"] == "assistant":
                    completion_parts.append(msg["content"])
            input_text = " ".join(prompt_parts)
            output_text = " ".join(completion_parts)
        elif "input" in sample and "output" in sample:
            input_text = sample["input"]
            output_text = sample["output"]
        else:
            return None

        prompt_str = self.LLAMA2_CHAT_TEMPLATE.format(input=input_text, output="")
        prompt_ids = self._tokenize(prompt_str)
        completion_ids = self._tokenize(output_text)
        eos_id = self._get_eos_id()

        input_ids = prompt_ids + completion_ids + [eos_id]
        loss_mask = [0] * len(prompt_ids) + [1] * len(completion_ids) + [0]

        if len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            loss_mask = loss_mask[: self.seq_length]

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "token_count": len(input_ids),
        }

    def _pack_next(self) -> Optional[Dict[str, list]]:
        """Pack multiple samples into one fixed-length sequence.

        Tracks sample boundaries via ``seq_start_id`` — a list of cumulative
        token offsets marking where each packed sample begins.  This enables
        proper cross-sample attention masking when used with flash-attention
        ``cu_seqlens`` APIs.
        """
        required = self.seq_length + 1
        all_ids: List[int] = []
        all_mask: List[int] = []
        seq_start_id: List[int] = [0]
        total = 0

        while total < required:
            if self._raw_idx >= len(self._raw_samples):
                if total == 0:
                    return None
                break
            sample = self._raw_samples[self._raw_idx]
            self._raw_idx += 1
            processed = self._process_sample(sample)
            if processed is None:
                continue
            all_ids.extend(processed["input_ids"])
            all_mask.extend(processed["loss_mask"])
            total += processed["token_count"]
            seq_start_id.append(total)

        eos_id = self._get_eos_id()
        while len(all_ids) < required:
            all_ids.append(eos_id)
            all_mask.append(0)

        seq_start_id = [min(s, required) for s in seq_start_id]
        if seq_start_id[-1] != required:
            seq_start_id.append(required)

        return {
            "input_ids": all_ids[:required],
            "loss_mask": all_mask[:required],
            "seq_start_id": seq_start_id,
        }
