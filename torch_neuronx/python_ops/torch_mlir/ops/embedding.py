"""Embedding operations."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::one_hot"],
    static_argnums=(1,),
    uses_preprocessing=True,
)
def torch_one_hot(x, num_classes=-1):
    def torch_one_hot_fn(x, num_classes):
        # Create output tensor with zeros
        output_shape = [*list(x.shape), num_classes]
        result = torch.zeros(output_shape, dtype=torch.int64, device=x.device)

        # Flatten input for scatter operation
        x_flat = x.flatten()

        # Reshape result for scatter and then reshape back
        result_flat = result.view(-1, num_classes)
        result_flat = torch.scatter(result_flat, 1, x_flat.unsqueeze(1), 1)

        return result_flat.view(output_shape)

    # Check dtype - must be int64 to match PyTorch behavior.
    if x.dtype != torch.int64:
        raise RuntimeError(
            f"one_hot is only applicable to index tensor of dtype int64, got {x.dtype}"
        )

    max_index = torch.max(x).item()
    # If num_classes is -1, infer it from data
    if num_classes < 0:
        num_classes = max_index + 1

    # Check for out-of-range indices
    if max_index >= num_classes:
        raise RuntimeError("Class values must be smaller than num_classes.")

    # Check for negative indices
    min_index = torch.min(x).item()
    if min_index < 0:
        raise RuntimeError("Class values must be non-negative.")

    return torch_one_hot_fn, (x, num_classes), {}
