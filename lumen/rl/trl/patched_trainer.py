###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GRPOTrainer subclass with stage-level timing injection.

Wraps TRL's GRPOTrainer to insert cuda-synchronized timing marks at
known boundaries inside the training loop: after generation, after
forward passes, after backward, and at step end. These marks are
consumed by GRPOPerfCallback to produce the stage breakdown.

This is spec §6.2 Option A (preferred): subclass with timing injection.
"""

import torch
from trl import GRPOTrainer

__all__ = ["PatchedGRPOTrainer"]


class PatchedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with stage-level timing marks for benchmarking.

    Overrides the training_step method to wrap TRL's internal logic
    with timing instrumentation. The perf_callback is discovered from
    the trainer's callback list at the first training step.
    """

    _perf_cb = None

    def _find_perf_callback(self):
        if self._perf_cb is not None:
            return self._perf_cb
        from lumen.rl.trl.perf_callback import GRPOPerfCallback
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, GRPOPerfCallback):
                self._perf_cb = cb
                return cb
        return None

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to inject timing marks around the training step.

        Marks step_start before the step, wraps accelerator.backward
        for backward timing, and relies on on_log for step_end.
        """
        cb = self._find_perf_callback()
        if cb is not None:
            cb.mark_step_start()

        return self.training_step_with_backward(model, inputs, num_items_in_batch)

    def _generate_and_score_completions(self, inputs):
        """Wrap TRL's generation + scoring to capture timing marks.

        TRL calls this method to:
          1. model.generate() — rollout
          2. forward passes for actor/reference log-probs
        We mark generation_end after generate returns and forward_end
        after the full method returns (which includes forward passes).

        If TRL renames this method in a future version, the parent's
        AttributeError will propagate — catch and degrade gracefully.
        """
        cb = self._find_perf_callback()

        model_for_gen = getattr(self, "model", None)
        if model_for_gen is None:
            return super()._generate_and_score_completions(inputs)

        original_generate = model_for_gen.generate

        def timed_generate(*args, **kwargs):
            result = original_generate(*args, **kwargs)
            if cb is not None:
                cb.mark_generation_end()
            return result

        model_for_gen.generate = timed_generate
        try:
            result = super()._generate_and_score_completions(inputs)
        finally:
            model_for_gen.generate = original_generate

        if cb is not None:
            cb.mark_forward_end()

        return result

    def training_step_with_backward(self, model, inputs, num_items_in_batch=None):
        """Override training_step to capture backward timing reliably.

        HF Trainer / Accelerate calls accelerator.backward(loss), not
        loss.backward() directly. We wrap the accelerator's backward
        method to ensure the timing mark fires regardless of the path.
        """
        cb = self._find_perf_callback()

        if cb is not None and hasattr(self, "accelerator"):
            original_backward = self.accelerator.backward

            def timed_backward(loss, **kwargs):
                original_backward(loss, **kwargs)
                cb.mark_backward_end()

            self.accelerator.backward = timed_backward
            try:
                result = super().training_step(model, inputs, num_items_in_batch)
            finally:
                self.accelerator.backward = original_backward
        else:
            result = super().training_step(model, inputs, num_items_in_batch)

        return result
