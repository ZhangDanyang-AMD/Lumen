#!/usr/bin/env python3
"""LLaMA2 pretraining entry point for DeepSpeed + Lumen."""

import logging

from lumen.models.llama2.deepspeed import DeepSpeedTrainer, get_args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    DeepSpeedTrainer(get_args()).train()


if __name__ == "__main__":
    main()
