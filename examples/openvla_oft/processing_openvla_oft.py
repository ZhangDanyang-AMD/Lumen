"""OpenVLA-OFT inputs and outputs: prompt tokens, image preprocessing, action (de)tokenization."""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from modeling_openvla_oft import (
    ACTION_DIM,
    ACTION_PLACEHOLDER_ID,
    ACTION_VOCAB_BEGIN,
    EMPTY_TOKEN_ID,
    IMAGE_SIZE,
    NUM_ACTION_TOKENS,
    NUM_ACTIONS_CHUNK,
    NUM_BINS,
    STOP_TOKEN_ID,
)

DINO_MEAN = (0.484375, 0.455078125, 0.40625)
DINO_STD = (0.228515625, 0.2236328125, 0.224609375)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class OpenVLAOFTProcessor:
    """Prompt tokenization, image preprocessing and action (de)tokenization."""

    def __init__(self, ckpt_dir: str, unnorm_key: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=False)
        with open(os.path.join(ckpt_dir, "dataset_statistics.json")) as f:
            stats = json.load(f)
        if unnorm_key not in stats:
            raise KeyError(f"unnorm_key {unnorm_key!r} not in dataset_statistics ({list(stats)})")
        act = stats[unnorm_key]["action"]
        self.q01 = torch.tensor(act["q01"], dtype=torch.float32)
        self.q99 = torch.tensor(act["q99"], dtype=torch.float32)
        self.norm_mask = torch.tensor(act["mask"], dtype=torch.bool)
        bins = torch.linspace(-1.0, 1.0, NUM_BINS)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self._prompt_cache: dict[str, list[int]] = {}
        self.num_action_tokens = NUM_ACTION_TOKENS

    @staticmethod
    def prompt_text(instruction: str) -> str:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def prompt_ids(self, instruction: str) -> list[int]:
        ids = self._prompt_cache.get(instruction)
        if ids is None:
            ids = self.tokenizer(self.prompt_text(instruction))["input_ids"]
            # Training sequences have the empty SentencePiece token between
            # "Out:" and the actions; without it every action after the first
            # in the chunk degenerates.
            if ids[-1] != EMPTY_TOKEN_ID:
                ids = ids + [EMPTY_TOKEN_ID]
            self._prompt_cache[instruction] = ids
        return ids

    def build_text_batch(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """``[prompt, 56 placeholders, STOP]`` per row, right-padded."""
        rows = [
            self.prompt_ids(s) + [ACTION_PLACEHOLDER_ID] * NUM_ACTION_TOKENS + [STOP_TOKEN_ID]
            for s in instructions
        ]
        width = max(len(r) for r in rows)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return ids, mask

    @staticmethod
    def preprocess_images(images: torch.Tensor) -> torch.Tensor:
        """uint8 ``[B, H, W, 3]`` -> float ``[B, 6, 224, 224]`` (DINO | SigLIP)."""
        x = images.permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", antialias=True, align_corners=False)
            x = x.clamp_(0.0, 1.0)
        dmean = x.new_tensor(DINO_MEAN).view(1, 3, 1, 1)
        dstd = x.new_tensor(DINO_STD).view(1, 3, 1, 1)
        smean = x.new_tensor(SIGLIP_MEAN).view(1, 3, 1, 1)
        sstd = x.new_tensor(SIGLIP_STD).view(1, 3, 1, 1)
        return torch.cat([(x - dmean) / dstd, (x - smean) / sstd], dim=1)

    def tokens_to_normalized(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Bin indices ``[B, 56]`` -> normalized actions ``[B, 8, 7]`` in ``[-1, 1]``."""
        vocab_ids = action_tokens.long() + ACTION_VOCAB_BEGIN
        discretized = (32000 - vocab_ids - 1).clamp(0, self.bin_centers.numel() - 1)
        centers = self.bin_centers.to(action_tokens.device)
        return centers[discretized].view(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)

    def unnormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        q01 = self.q01.to(normalized.device)
        q99 = self.q99.to(normalized.device)
        mask = self.norm_mask.to(normalized.device)
        raw = 0.5 * (normalized + 1.0) * (q99 - q01 + 1e-8) + q01
        return torch.where(mask, raw, normalized)

    def tokens_to_env_actions(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Bin indices -> LIBERO env actions ``[B, 8, 7]``.

        The policy's gripper output is the dataset convention (1 = open);
        LIBERO expects -1 = open, +1 = close.
        """
        actions = self.unnormalize(self.tokens_to_normalized(action_tokens))
        gripper = torch.where(actions[..., -1] > 0.5, -1.0, 1.0)
        return torch.cat([actions[..., :-1], gripper.unsqueeze(-1)], dim=-1)
