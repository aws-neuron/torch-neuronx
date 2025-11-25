"""No-cast ArgumentProcessor to avoid implicit dtype conversions.

This processor ensures tensor inputs are contiguous but does not perform any
dtype conversions in preprocessing. It is useful for ops like device-side
casting or copies where dtype is controlled explicitly by the op/kernel and
we want to avoid redispatch or accidental pre-casts.
"""

from __future__ import annotations

import torch

from .argument_processor import ArgumentProcessor


class NoCastArgumentProcessor(ArgumentProcessor):
    """Argument processor that avoids dtype conversions for tensors.

    - Preserves tensor dtypes
    - Ensures contiguity for predictable kernel memory access
    - Respects static_argnums/static_argnames like the base class
    """

    def preprocess_inputs(self, inputs: tuple) -> tuple:
        converted = []
        for i, inp in enumerate(inputs):
            if i in self.static_argnums:
                converted.append(inp)
            elif isinstance(inp, torch.Tensor):
                converted.append(inp if inp.is_contiguous() else inp.contiguous())
            else:
                converted.append(inp)
        return tuple(converted)
