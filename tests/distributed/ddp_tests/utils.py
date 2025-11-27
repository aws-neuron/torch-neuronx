import torch


def _get_parameter_sizes(param_list):
    """Get parameter sizes in bytes.

    Args:
        param_list: List of model parameters.

    Returns:
        List of parameter sizes in bytes.
    """
    return [p.numel() * p.element_size() for p in param_list]


def _build_buckets(param_sizes, bucket_cap_bytes):
    """Build buckets based on parameter sizes and capacity.

    Args:
        param_sizes: List of parameter sizes in bytes.
        bucket_cap_bytes: Maximum bucket capacity in bytes.

    Returns:
        tuple: (buckets, bucket_sizes) where buckets contains parameter indices
               and bucket_sizes contains the size of each bucket.
    """
    buckets = []
    current_bucket = []
    current_size = 0
    bucket_sizes = []

    for i in range(len(param_sizes)):
        current_bucket.append(len(param_sizes) - 1 - i)  # Append indices in reverse order
        current_size += param_sizes[i]

        if current_size > bucket_cap_bytes:
            bucket_sizes.append(current_size)
            current_size = 0
            buckets.append(current_bucket)  # mark this bucket as closed
            current_bucket = []  # start a new bucket

    if current_bucket:
        buckets.append(current_bucket)
        bucket_sizes.append(current_size)

    return buckets, bucket_sizes


def compute_ddp_buckets_from_ddp(
    ddp_model: torch.nn.parallel.DistributedDataParallel, bucket_cap_bytes: int
):
    """Compute DDP-style rebuilt buckets for a DDP-wrapped model.

    Parameters:
        ddp_model: The DDP-wrapped model to analyze.
        bucket_cap_bytes: Maximum bucket capacity in bytes.

    Returns:
        tuple: A tuple containing (buckets, bucket_sizes) where buckets is a list
               of parameter indices and bucket_sizes is a list of bucket sizes in bytes.
    """
    # Extract the underlying module
    model = ddp_model.module

    # Get all parameters in reverse order for DDP rebuilt bucket logic
    param_list = list(model.parameters())
    param_list.reverse()

    # Compute parameter sizes and build buckets
    param_sizes = _get_parameter_sizes(param_list)
    return _build_buckets(param_sizes, bucket_cap_bytes)


def compare_ddp_buckets(ddp_logging_data, model_buckets, bucket_sizes):
    """Compare computed DDP buckets with actual DDP logging data.

    Parameters:
        ddp_logging_data: Dictionary containing DDP logging information.
        model_buckets: List of computed bucket parameter indices.
        bucket_sizes: List of computed bucket sizes in bytes.

    Raises:
        AssertionError: If computed buckets don't match DDP logging data.
    """
    # Parse logging data
    log_sizes = [int(s.strip()) for s in ddp_logging_data["rebuilt_bucket_sizes"].split(",")]
    log_indices = [
        [int(i) for i in group.strip().split()]
        for group in ddp_logging_data["rebuilt_per_bucket_param_indices"].split(",")
    ]

    assert ddp_logging_data["num_buckets_reduced"] == len(model_buckets), (
        f"Number of buckets mismatch: DDP logged {ddp_logging_data['num_buckets_reduced']}, "
        f"but computed {len(model_buckets)}"
    )
    assert (
        log_sizes == bucket_sizes
    ), f"Bucket sizes mismatch: DDP logged {log_sizes}, but computed {bucket_sizes}"
    assert (
        model_buckets == log_indices
    ), f"Bucket indices mismatch: DDP logged {log_indices}, but computed {model_buckets}"
