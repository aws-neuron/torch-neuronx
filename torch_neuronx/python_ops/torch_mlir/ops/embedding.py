"""Embedding operations."""

import torch

from torch_neuronx.python_ops.nki_kernels.bound_check import assert_index_in_bounds

from ..operation_registry import Preprocessing, register_aten


def _validate_one_hot(x, num_classes=-1):
    """Validate one_hot dtypes on raw args."""
    if x.dtype != torch.int64:
        raise RuntimeError(
            f"one_hot is only applicable to index tensor of dtype int64, got {x.dtype}"
        )


def _one_hot_error_handler(ctx):
    """Raise one_hot's exact class-value exception on an OOB device fault (see
    op_error_handlers.py). The compute captures (x, num_classes) positionally,
    so x is ctx.tensor(0). Negative values report first, matching the eager
    check order."""
    x = ctx.tensor(0)
    if int(torch.min(x)) < 0:
        raise RuntimeError("Class values must be non-negative.")
    raise RuntimeError("Class values must be smaller than num_classes.")


@register_aten(
    ["aten::one_hot"],
    static_argnums=(1,),
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    validate_static_inputs=_validate_one_hot,
    # num_classes = max(x) + 1 (when -1) is value-derived, so re-lower every call.
    relower_every_call=True,
    error_handler=_one_hot_error_handler,
)
def torch_one_hot(x, num_classes=-1):
    def torch_one_hot_fn(x, num_classes):
        # Create output tensor with zeros
        output_shape = [*list(x.shape), num_classes]
        result = torch.zeros(output_shape, dtype=torch.int64, device=x.device)

        # Flatten input for scatter operation
        x_flat = x.flatten()

        # In-graph bounds check (no host sync): class values must be in
        # [0, num_classes). The checked index feeds the scatter (survives DCE)
        # and faults on an out-of-range class value; _one_hot_error_handler then
        # surfaces the exact message.
        x_flat = assert_index_in_bounds(x_flat, num_classes, lo=0)

        # Reshape result for scatter and then reshape back
        result_flat = result.view(-1, num_classes)
        result_flat = torch.scatter(result_flat, 1, x_flat.unsqueeze(1), 1)

        return result_flat.view(output_shape)

    # num_classes = max(x) + 1 (when -1) sets the output shape, so a host read is
    # unavoidable here; the value *bounds* check itself is done in-graph above.
    if num_classes < 0:
        num_classes = int(torch.max(x)) + 1

    return torch_one_hot_fn, (x, num_classes), {}
