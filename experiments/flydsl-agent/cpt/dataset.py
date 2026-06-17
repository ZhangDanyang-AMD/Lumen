"""CPT dataset with per-document weighted sampling.

Reads JSONL with ``{"text": "...", "meta": {"weight": 5.4, ...}}`` records,
tokenizes and concatenates documents with EOS separators, then chunks into
fixed-length sequences for next-token prediction.

Each chunk inherits the weight of the document it came from, enabling
``WeightedRandomSampler`` to oversample high-priority content (expert skills,
gold kernels) during training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["CPTDataset"]

logger = logging.getLogger(__name__)


class CPTDataset(Dataset):
    """Continued pre-training dataset with weighted sampling support.

    Args:
        data_path: Path to a ``.jsonl`` file with ``text`` and ``meta.weight``
            fields (the format produced by the flydsl-agent data pipeline).
        seq_length: Context window in tokens.  Each sample has ``seq_length + 1``
            tokens (the extra token provides the label for the last position).
        tokenizer: Any object with ``encode(text) -> list[int]`` and
            ``eos_token_id``.
        max_samples: Cap ``__len__`` if set.
        default_weight: Fallback weight for records missing ``meta.weight``.
    """

    def __init__(
        self,
        data_path: Optional[str],
        seq_length: int,
        tokenizer,
        max_samples: Optional[int] = None,
        default_weight: float = 1.0,
    ):
        self.seq_length = seq_length
        self._chunks: List[List[int]] = []
        self._chunk_weights: List[float] = []
        self._max_samples = max_samples

        if data_path is None:
            return

        records = self._load_jsonl(data_path)
        self._build_chunks(records, tokenizer, default_weight)

        logger.info(
            "CPTDataset: %d chunks (seq_length=%d) from %d documents in %s",
            len(self._chunks),
            seq_length,
            len(records),
            data_path,
        )

    @staticmethod
    def _load_jsonl(path: str) -> List[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _build_chunks(
        self,
        records: List[dict],
        tokenizer,
        default_weight: float,
    ) -> None:
        chunk_len = self.seq_length + 1
        eos_id = tokenizer.eos_token_id

        for record in records:
            text = record.get("text", "")
            if not text:
                continue

            weight = default_weight
            meta = record.get("meta", {})
            if isinstance(meta, dict):
                weight = meta.get("weight", default_weight)

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if eos_id is not None:
                token_ids.append(eos_id)

            n_chunks = len(token_ids) // chunk_len
            for i in range(n_chunks):
                self._chunks.append(token_ids[i * chunk_len : (i + 1) * chunk_len])
                self._chunk_weights.append(weight)

    def __len__(self) -> int:
        n = len(self._chunks)
        if self._max_samples is not None:
            return min(n, self._max_samples)
        return n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = idx % len(self._chunks)
        chunk = self._chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    @property
    def weights(self) -> List[float]:
        """Per-sample weights for ``WeightedRandomSampler``."""
        n = len(self)
        return self._chunk_weights[:n]

    def token_count(self) -> int:
        """Total tokens across all chunks (for epoch/step estimation)."""
        return len(self._chunks) * self.seq_length
