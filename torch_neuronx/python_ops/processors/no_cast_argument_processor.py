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
    """Prepare arguments without converting tensor dtypes.

    Args:
        static_argnames (tuple[str, ...], optional): Keyword argument names to keep
            static. Defaults to ``None``.

    Note:
        Tensor dtypes are preserved, noncontiguous tensors are made contiguous,
        and inherited static positional and keyword arguments remain unchanged.
    """

    def preprocess_inputs(self, inputs: tuple) -> tuple:
        """Prepare inputs without converting tensor dtypes.

        Args:
            inputs (tuple): Input tensors, dtypes, scalars, and nested structures.

        Returns:
            tuple: Inputs with noncontiguous tensors made contiguous and nested
            structures converted to tuples.

        Raises:
            RuntimeError: If static positional indices have not been configured.

        Note:
            ``torch.dtype`` values become scalar-type integers while tensor dtypes are
            preserved. Inputs at static argument indices remain unchanged.
        """
        converted = []
        for i, inp in enumerate(inputs):
            if i in self.static_argnums:
                converted.append(inp)
            elif isinstance(inp, torch.Tensor):
                converted.append(inp if inp.is_contiguous() else inp.contiguous())
            # Handle nested structures
            elif isinstance(inp, (list | tuple)):
                converted.append(self.preprocess_inputs(tuple(inp)))
            elif isinstance(inp, torch.dtype):
                # Map torch.dtype to ScalarType enum values as defined in
                # c10/core/ScalarType.h for MLIR lowring
                scalar_type_int = ArgumentProcessor._get_scalar_type_value(inp)
                converted.append(scalar_type_int)
            else:
                converted.append(inp)

        return tuple(converted)
