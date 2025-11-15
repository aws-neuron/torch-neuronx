import jax
import jax.numpy as jnp
import torch


# @register_aten(
#     ["aten::cross_entropy_loss"],
#     static_argnums=(3, 4, 5),
#     uses_preprocessing=True,
# )
def _aten_cross_entropy_loss(
    input, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    # Validation function for preprocessing
    def _validate_inputs(input, target, weight, ignore_index):
        input_shape = input.shape

        # Check spatial dimension requirements
        if len(input_shape) > 2:
            is_prob_target = target.dim() == input.dim()

            if is_prob_target:
                # Probability target should have same shape as input
                if target.shape != input.shape:
                    raise RuntimeError(
                        f"size mismatch (got input: {list(input_shape)} ,"
                        f" target: {list(target.shape)})"
                    )
            else:
                expected_target_dims = len(input_shape) - 1
                if target.dim() != expected_target_dims:
                    raise RuntimeError(
                        f"only batches of spatial targets supported ({expected_target_dims}D "
                        f"tensors) but got targets of dimension: {target.dim()}"
                    )
                for i in range(2, len(input_shape)):
                    if input_shape[i] != target.shape[i - 1]:
                        # Match PyTorch's error message format
                        raise RuntimeError(
                            f"size mismatch (got input: {list(input_shape)} ,"
                            f" target: {list(target.shape)})"
                        )

        n_classes = input_shape[1] if len(input_shape) > 2 else input_shape[-1]

        # Validate target values for class index targets
        if target.dim() != input.dim():
            valid_indices = target != ignore_index
            if torch.any(valid_indices):
                valid_targets = torch.masked_select(target, valid_indices)

                max_idx = torch.max(valid_targets)
                if max_idx >= n_classes:
                    raise IndexError(f"Target {max_idx.item()} is out of bounds.")

                min_idx = torch.min(valid_targets)
                if min_idx < 0:
                    raise IndexError(f"Target {min_idx.item()} is out of bounds.")

        if weight is not None and len(weight) != n_classes:
            shape_str = f"[{weight.size(0)}]" if weight.dim() == 1 else str(list(weight.shape))
            raise RuntimeError(
                f"weight tensor should be defined either for all {n_classes} classes or"
                f" no classes but got weight tensor of shape: {shape_str}"
            )

    _validate_inputs(input, target, weight, ignore_index)

    def _cross_entropy_loss_fn(
        input, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
    ):
        input_shape = input.shape
        target_shape = target.shape

        # Helper function for computing loss with probability targets
        def compute_prob_target_loss(log_prob, target, n_classes, num_samples):
            if label_smoothing > 0:
                uniform = jnp.ones_like(target) / n_classes
                target = (1.0 - label_smoothing) * target + label_smoothing * uniform

            if weight is not None:
                weight_reshaped = jnp.reshape(weight, (1, -1))
                weighted_log_prob = log_prob * weight_reshaped
                loss = -jnp.sum(target * weighted_log_prob, axis=-1)
            else:
                loss = -jnp.sum(target * log_prob, axis=-1)

            weight_sum = jnp.array(num_samples, dtype=jnp.float32)
            return loss, weight_sum

        # Helper function for computing loss with class index targets
        def compute_class_target_loss(log_prob, target, n_classes):
            valid_mask = target != ignore_index
            valid_targets = jnp.where(valid_mask, target, 0)

            target_ohe = jax.nn.one_hot(valid_targets, n_classes)
            mask_expanded = jnp.expand_dims(valid_mask, axis=-1)
            target_ohe = target_ohe * mask_expanded

            if label_smoothing > 0:
                target_for_loss = (
                    1.0 - label_smoothing
                ) * target_ohe + label_smoothing / n_classes * mask_expanded
            else:
                target_for_loss = target_ohe

            if weight is not None:
                weight_reshaped = jnp.reshape(weight, (1, -1))
                weighted_log_prob = log_prob * weight_reshaped
                loss = -jnp.sum(target_for_loss * weighted_log_prob, axis=-1)

                if label_smoothing > 0:
                    target_weights = jnp.sum(target_for_loss * weight_reshaped, axis=-1)
                else:
                    target_weights = jnp.sum(target_ohe * weight_reshaped, axis=-1)
                weight_sum = jnp.sum(target_weights)
            else:
                loss = -jnp.sum(target_for_loss * log_prob, axis=-1)
                weight_sum = jnp.sum(valid_mask.astype(jnp.float32))

            return loss, weight_sum

        # Handle spatial inputs (more than 2D)
        if len(input_shape) > 2:
            n_classes = input_shape[1]
            is_prob_target = target.ndim == input.ndim

            perm = [0, *list(range(2, len(input_shape))), 1]
            input_transposed = jnp.transpose(input, perm)

            input_flattened = jnp.reshape(input_transposed, (-1, n_classes))

            if is_prob_target:
                target_transposed = jnp.transpose(target, perm)
                target_flattened = jnp.reshape(target_transposed, (-1, n_classes))
            else:
                target_flattened = jnp.reshape(target, (-1,))

            flat_shape = input_flattened.shape
            flat_n_classes = flat_shape[-1]
            flat_is_prob = target_flattened.ndim == input_flattened.ndim
            flat_num_samples = flat_shape[0]

            flat_log_prob = jax.nn.log_softmax(input_flattened, axis=-1)

            if flat_is_prob:
                flat_loss, flat_weight_sum = compute_prob_target_loss(
                    flat_log_prob, target_flattened, flat_n_classes, flat_num_samples
                )
            else:
                flat_loss, flat_weight_sum = compute_class_target_loss(
                    flat_log_prob, target_flattened, flat_n_classes
                )

            # 0 is none, 2 is sum
            if reduction == 0:
                output_shape = (
                    target_shape if not is_prob_target else target_shape[0:1] + target_shape[2:]
                )
                return jnp.reshape(flat_loss, output_shape)
            elif reduction == 2:
                return jnp.sum(flat_loss)
            else:
                return jnp.sum(flat_loss) / flat_weight_sum
        else:
            n_classes = input_shape[-1]
            is_prob_target = target.ndim == input.ndim

            num_samples = 1 if len(input_shape) == 1 else input_shape[0]

            log_prob = jax.nn.log_softmax(input, axis=-1)

            if is_prob_target:
                loss, weight_sum = compute_prob_target_loss(
                    log_prob, target, n_classes, num_samples
                )
            else:
                loss, weight_sum = compute_class_target_loss(log_prob, target, n_classes)

            if reduction == 0:
                return loss
            elif reduction == 2:
                return jnp.sum(loss)
            else:
                return jnp.sum(loss) / weight_sum

    return (
        _cross_entropy_loss_fn,
        (input, target, weight, reduction, ignore_index, label_smoothing),
        {},
    )
