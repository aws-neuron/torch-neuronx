# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
import logging
import os
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int]:
    """Initialize a Gloo process group from environment variables.

    Returns:
        tuple[int, int]: Current process rank and world size.

    Raises:
        KeyError: If ``RANK`` or ``WORLD_SIZE`` is absent from the environment.
        ValueError: If either environment value cannot be converted to an integer.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    return rank, world_size


def log_model_info(rank: int, model: torch.nn.Module, model_name: str):
    """Log parameter statistics for a model.

    Args:
        rank (int): Process rank included in log messages.
        model (torch.nn.Module): Model whose parameters to count.
        model_name (str): Model name included in log messages.

    Returns:
        None: This function only emits log messages.
    """
    param_count = sum(p.numel() for p in model.parameters())
    param_nonzero = sum((p != 0).sum().item() for p in model.parameters())
    logger.info(f"Rank {rank}: Model: {model_name}")
    logger.info(f"Rank {rank}: Parameters - total: {param_count}, non-zero: {param_nonzero}")


def run_eager_forward(rank: int, model: torch.nn.Module, *inputs) -> tuple[torch.Tensor, float]:
    """Run an eager forward pass on a Neuron device.

    Args:
        rank (int): Index of the target Neuron device.
        model (torch.nn.Module): Model to move to the device and execute.
        *inputs (Any): Inputs to localize when needed and move to the device.

    Returns:
        tuple[torch.Tensor, float]: Model output and wall-clock model-call time in seconds.

    Note:
        ``model.to(device)`` moves the model parameters and buffers in place.
    """
    device = torch.device(f"neuron:{rank}")

    # Move model to neuron if not already
    model = model.to(device)

    # Move inputs to neuron device
    neuron_inputs = []
    for inp in inputs:
        if hasattr(inp, "to_local"):
            # Handle DTensor - move local tensor to neuron
            local_tensor = inp.to_local().to(device)
            neuron_inputs.append(local_tensor)
        elif isinstance(inp, torch.Tensor):
            neuron_inputs.append(inp.to(device))
        else:
            neuron_inputs.append(inp)

    logger.info(f"Rank {rank}: Running eager forward pass on neuron...")
    eager_start = time.time()

    with torch.no_grad():
        eager_output = model(*neuron_inputs)

    eager_time = time.time() - eager_start
    logger.info(f"Rank {rank}: Eager forward: {eager_time:.2f}s")
    logger.info(f"Rank {rank}: Eager output shape: {eager_output.shape}")
    logger.info(
        f"Rank {rank}: Eager output stats - mean: {eager_output.mean():.6f}, "
        + f"std: {eager_output.std():.6f}"
    )

    return eager_output, eager_time


def compile_model(rank: int, model: torch.nn.Module, model_name: str):
    # Runs torch.compile with the neuron backend.
    # Assumes model is already on neuron device
    """Wrap a model with the Dynamo backend.

    Args:
        rank (int): Process rank appended to the registered model name.
        model (torch.nn.Module): Model to compile; it must already be on a Neuron device.
        model_name (str): Base name registered for the compilation.

    Returns:
        torch.nn.Module: Full-graph compiled model wrapper.
    """
    from torch_neuronx.neuron_dynamo_backend import set_model_name

    set_model_name(f"{model_name}_rank{rank}")

    logger.info(f"Rank {rank}: Starting Neuron compilation...")
    compile_start = time.time()

    compiled_model = torch.compile(model, backend="neuron", fullgraph=True)

    compile_time = time.time() - compile_start
    logger.info(f"Rank {rank}: Compilation setup: {compile_time:.2f}s")

    return compiled_model


def run_neuron_forward(
    rank: int, compiled_model: torch.nn.Module, *inputs, run_number: int = 1
) -> tuple[torch.Tensor, float]:
    # Neuron forward pass
    """Run a compiled forward pass on a Neuron device.

    Args:
        rank (int): Index of the target Neuron device.
        compiled_model (torch.nn.Module): Compiled model to execute.
        *inputs (Any): Inputs to localize when needed and move to the device.
        run_number (int, optional): Run number used in log messages. Defaults to ``1``.

    Returns:
        tuple[torch.Tensor, float]: Model output and wall-clock model-call time in seconds.
    """
    if run_number == 1:
        logger.info(f"Rank {rank}: Running first Neuron forward pass (compiles NEFF)...")
    else:
        logger.info(f"Rank {rank}: Running Neuron forward pass #{run_number}...")

    # Move inputs to neuron device
    device = torch.device(f"neuron:{rank}")
    neuron_inputs = []
    for inp in inputs:
        if hasattr(inp, "to_local"):
            # Handle DTensor - move local tensor to neuron
            local_tensor = inp.to_local().to(device)
            neuron_inputs.append(local_tensor)
        elif isinstance(inp, torch.Tensor):
            neuron_inputs.append(inp.to(device))
        else:
            neuron_inputs.append(inp)

    forward_start = time.time()

    with torch.no_grad():
        neuron_output = compiled_model(*neuron_inputs)

    forward_time = time.time() - forward_start

    if run_number == 1:
        logger.info(f"Rank {rank}: First Neuron forward (with compilation): {forward_time:.2f}s")
    else:
        logger.info(f"Rank {rank}: Neuron forward #{run_number}: {forward_time:.2f}s")

    logger.info(f"Rank {rank}: Neuron output shape: {neuron_output.shape}")
    logger.info(
        f"Rank {rank}: Neuron output stats - mean: {neuron_output.mean():.6f}, "
        + f"std: {neuron_output.std():.6f}"
    )

    return neuron_output, forward_time


def log_comparison_summary(rank: int, model_name: str):
    # Prints that were in the previous tests
    """Log the comparison completion summary on rank zero.

    Args:
        rank (int): Process rank; nonzero ranks do not log the summary.
        model_name (str): Model name displayed in the summary.

    Returns:
        None: This function only emits log messages.
    """
    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info(f"{model_name.upper()} ACCURACY TEST COMPLETE")
        logger.info("=" * 80)
        logger.info("Model compiled successfully")
        logger.info("CPU and Neuron forward passes executed")
        logger.info("Accuracy validation completed")
        logger.info("=" * 80)


def check_neuron_consistency(
    rank: int, output1: torch.Tensor, output2: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6
) -> bool:
    # Check if neuron tests were the same across runs
    """Check whether two Neuron outputs are numerically close.

    Args:
        rank (int): Process rank included in the log message.
        output1 (torch.Tensor): First output to compare.
        output2 (torch.Tensor): Second output to compare.
        atol (float, optional): Absolute tolerance. Defaults to ``1e-6``.
        rtol (float, optional): Relative tolerance. Defaults to ``1e-6``.

    Returns:
        bool: ``True`` when ``torch.allclose`` accepts the outputs.
    """
    consistent = torch.allclose(output1, output2, atol=atol, rtol=rtol)
    logger.info(f"Rank {rank}: Neuron outputs consistent across runs: {consistent}")
    return consistent
