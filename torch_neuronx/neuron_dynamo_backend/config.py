# ruff: noqa: SIM108

"""
Global configuration and state management for Neuron backend
"""

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import torch

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool, _getenv_int, _getenv_path

logger = logging.getLogger(__name__)

# TODO (NF-20): remove global variables
# Global variable to pass model name to compiler
_current_model_name = None

# Global variable to store current timestamp for this compilation
_current_timestamp = None


def get_rank() -> int:
    """Get the current process rank.

    Returns the distributed rank if torch.distributed is initialized,
    otherwise returns the current Neuron device ID.

    Returns:
        int: Process rank or device ID.
    """
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else torch.neuron.current_device()
    )


def get_local_rank() -> int:
    """Get the current process rank within the local node.

    Returns the node-local rank if torch.distributed is initialized,
    otherwise returns the current Neuron device ID.

    Returns:
        int: Local process rank or device ID.
    """
    return (
        torch.distributed.get_node_local_rank()
        if torch.distributed.is_initialized()
        else torch.neuron.current_device()
    )


def get_world_size() -> int:
    """Get the total number of processes in the distributed group.

    Returns the distributed world size if torch.distributed is initialized,
    otherwise returns the number of available Neuron devices.

    Returns:
        int: Number of processes or devices.
    """
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else torch.neuron.device_count()
    )


def get_local_world_size() -> int:
    """Get the number of processes within the local node.

    Reads from LOCAL_WORLD_SIZE environment variable if set,
    otherwise returns the number of available Neuron devices.

    Returns:
        int: Number of local processes or devices.
    """
    return _getenv_int("LOCAL_WORLD_SIZE", torch.neuron.device_count())


def get_current_timestamp() -> str:
    """Get or create timestamp for current compilation

    Returns:
        str: Same timestamp for all artifacts in a single compilation.
             Uses microsecond precision to handle graph breaks that occur in quick succession.
    """
    global _current_timestamp
    if _current_timestamp is None:
        _current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return _current_timestamp


def reset_timestamp():
    """Reset the timestamp (called at start of new compilation)"""
    global _current_timestamp
    _current_timestamp = None


@contextmanager
def managed_artifacts_directory():
    """Context manager for artifact directory management.

    Gets the artifacts directory and ensures cleanup if
    TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS is False.

    Yields:
        Path: The artifacts directory path
    """
    preserve_artifacts = _getenv_bool("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", False)
    temp_dir = get_artifacts_directory()

    try:
        yield temp_dir, preserve_artifacts
    finally:
        cleanup_compilation_artifacts(temp_dir, preserve=preserve_artifacts)


@lru_cache(maxsize=1)
def get_artifacts_directory() -> Path:
    """Get the artifacts directory from environment variable or default

    Priority:
    1. TORCH_NEURONX_DEBUG_DIR environment variable
    2. Default: /tmp/neuron_backend_<random>/

    Returns:
        Path: Base artifacts directory
    """
    debug_dir = _getenv_path("TORCH_NEURONX_DEBUG_DIR", None)
    if debug_dir is None:
        debug_dir = Path(tempfile.mkdtemp(prefix="neuron_backend_"))

    # Ensure directory exists
    os.makedirs(debug_dir, exist_ok=True)

    return debug_dir


def cleanup_compilation_artifacts(temp_dir: Path, preserve: bool = True):
    """Clean up temporary compilation artifacts

    Args:
        temp_dir: Temporary directory to clean up
        preserve: Whether to preserve artifacts
    """
    try:
        if temp_dir.exists():
            if preserve:
                logger.debug(f"Preserving compilation artifacts for debugging: {temp_dir}")
            else:
                logger.debug(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up {temp_dir}: {e}")


def get_fx_graph_path(model_name: str | None = None) -> Path:
    """Get the path for saving FX graph artifacts

    Structure: fx_graphs/proc_{rank}/{model_name}_{timestamp}.fx.txt

    Args:
        model_name: Optional model name, uses timestamp if not provided

    Returns:
        Path: Full path to FX graph file
    """
    base_dir = get_artifacts_directory()
    rank = get_rank()
    fx_dir = base_dir / "fx_graphs" / f"proc_{rank}"
    fx_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_current_timestamp()

    if model_name:
        filename = f"{model_name.lower()}_{timestamp}.fx.txt"
    else:
        filename = f"{timestamp}.fx.txt"

    return fx_dir / filename


def get_stablehlo_path(model_name: str | None = None) -> Path:
    """Get the path for saving StableHLO artifacts

    Structure: stablehlo/proc_{rank}/{model_name}_{timestamp}.stablehlo.mlir

    Args:
        model_name: Optional model name, uses timestamp if not provided

    Returns:
        Path: Full path to StableHLO file
    """
    base_dir = get_artifacts_directory()
    rank = get_rank()
    stablehlo_dir = base_dir / "stablehlo" / f"proc_{rank}"
    stablehlo_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_current_timestamp()

    if model_name:
        filename = f"{model_name.lower()}_{timestamp}.stablehlo.mlir"
    else:
        filename = f"{timestamp}.stablehlo.mlir"

    return stablehlo_dir / filename


def get_neuronx_cc_working_dir() -> Path:
    """Get the working directory path for neuronx-cc compiler invocation.

    Creates the directory structure: {artifacts_dir}/neuronx_cc/proc_{rank}/

    Returns:
        Path: Working directory path for neuronx-cc.
    """
    base_dir = get_artifacts_directory()
    rank = get_rank()
    neuronx_cc_dir = base_dir / "neuronx_cc" / f"proc_{rank}"
    neuronx_cc_dir.mkdir(parents=True, exist_ok=True)
    return neuronx_cc_dir


def get_neff_path(model_name: str | None = None) -> Path:
    """Get the path for saving NEFF artifacts

    Structure: neff/proc_{rank}/{model_name}_{timestamp}.neff

    Args:
        model_name: Optional model name, uses timestamp if not provided

    Returns:
        Path: Full path to NEFF file
    """
    base_dir = get_artifacts_directory()
    rank = get_rank()
    neff_dir = base_dir / "neff" / f"proc_{rank}"
    neff_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_current_timestamp()

    if model_name:
        filename = f"{model_name.lower()}_{timestamp}.neff"
    else:
        filename = f"{timestamp}.neff"

    return neff_dir / filename


def get_raw_torch_path(model_name: str | None = None) -> Path:
    """Get the path for saving RAW Torch IR artifacts.

    Structure: {artifacts_dir}/raw/proc_{rank}/{model_name}_{timestamp}.torch_raw.mlir

    Args:
        model_name (str | None): Model name for filename. Uses current model name if None.

    Returns:
        Path: Full path to RAW Torch IR file.
    """
    base_dir = get_artifacts_directory()
    rank = get_rank()
    raw_dir = base_dir / "raw" / f"proc_{rank}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = get_model_name()

    timestamp = get_current_timestamp()

    if model_name:
        filename = f"{model_name.lower()}_{timestamp}.torch_raw.mlir"
    else:
        filename = f"{timestamp}.torch_raw.mlir"

    return raw_dir / filename


def get_err_mlir_path() -> Path:
    """Get the path for saving MLIR error artifacts.

    Creates the directory structure: {artifacts_dir}/torch_mlir_error/proc_{rank}/

    Returns:
        Path: Full path to error MLIR file.
    """
    base_dir = get_artifacts_directory()
    rank = get_rank()
    err_dir = base_dir / "torch_mlir_error" / f"proc_{rank}"
    err_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_current_timestamp()
    filename = f"{get_model_name().lower()}_{timestamp}.mlir"
    return err_dir / filename


def get_transformed_fx_path(model_name: str | None = None) -> Path:
    """Get the path for saving transformed FX graph artifacts.

    Creates artifacts after collective transforms are applied.
    Structure:
        {artifacts_dir}/transformed_fx/proc_{rank}/{model_name}_{timestamp}.transformed.fx.txt

    Args:
        model_name (str | None): Model name for filename. Uses current model name if None.

    Returns:
        Path: Full path to transformed FX graph file.
    """
    base_dir = get_artifacts_directory()
    rank = get_rank()
    transformed_fx_dir = base_dir / "transformed_fx" / f"proc_{rank}"
    transformed_fx_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = get_model_name()

    timestamp = get_current_timestamp()

    if model_name:
        filename = f"{model_name.lower()}_{timestamp}.transformed.fx.txt"
    else:
        filename = f"{timestamp}.transformed.fx.txt"

    return transformed_fx_dir / filename


def set_model_name(name: str):
    """Set the current model name for NEFF file naming

    Args:
        name (str): Name for the module.
    """
    global _current_model_name
    _current_model_name = name
    logger.debug(f"Model name set to: {name}")


def get_model_name() -> str:
    """Get the current model name

    Returns:
        str: Name of current model
    """
    global _current_model_name
    if _current_model_name is None:
        _current_model_name = "model_default"
        logger.debug(f"Using default model name {_current_model_name}")
    return _current_model_name
