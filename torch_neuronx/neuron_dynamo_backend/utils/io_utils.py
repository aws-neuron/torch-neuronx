"""
I/O utilities for FX graphs and MLIR modules

This module provides utilities for saving and loading FX graphs and MLIR modules
in various formats for debugging and analysis purposes.
"""

import logging
import shutil
from pathlib import Path

import torch
from torch_mlir import fx
from torch_mlir import ir as mlir_ir
from torch_mlir.compiler_utils import OutputType

logger = logging.getLogger(__name__)


# ============================================================================
# FX Graph I/O
# ============================================================================


def save_fx_graph_txt(gm: torch.fx.GraphModule, path: str | Path) -> Path:
    """
    Save FX GraphModule as human-readable text file (.fx.txt)

    Args:
        gm: FX GraphModule to save
        path: Output path for .fx.txt file

    Returns:
        Path: Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get readable representation
    readable_graph = gm.print_readable(print_output=False)

    with open(path, "w") as f:
        f.write(readable_graph)

    logger.debug(f"Saved FX graph (text) to: {path}")

    return path


# ============================================================================
# MLIR I/O
# ============================================================================


def fx_to_mlir_string(
    gm: torch.fx.GraphModule,
    output_type: OutputType = OutputType.RAW,
    verbose: bool = False,
    enable_ir_printing: bool = False,
) -> str:
    """
    Convert FX GraphModule to MLIR string

    Args:
        gm: FX GraphModule to convert
        output_type: MLIR output type (RAW, STABLEHLO, etc.)
        verbose: Enable verbose output
        enable_ir_printing: Enable IR printing during conversion

    Returns:
        str: MLIR string representation
    """
    logger.debug(f"Converting FX graph to MLIR ({output_type})...")

    mlir_module = fx.stateless_fx_import(
        gm, output_type=output_type, verbose=verbose, enable_ir_printing=enable_ir_printing
    )

    mlir_str = str(mlir_module)
    logger.debug(f"Converted FX graph to MLIR ({output_type})")

    return mlir_str


def save_mlir(mlir_str: str, path: str | Path) -> Path:
    """
    Save MLIR string to file

    Args:
        mlir_str: MLIR string to save
        path: Output path (e.g., .raw.mlir, .stablehlo.mlir)

    Returns:
        Path: Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(mlir_str)

    logger.debug(f"Saved MLIR to: {path}")

    return path


def load_mlir(path: str | Path, context=None) -> mlir_ir.Module:
    """
    Load MLIR module from file

    Args:
        path: Path to MLIR file
        context: Optional MLIR context (creates new if None)

    Returns:
        mlir_ir.Module: Loaded MLIR module
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"MLIR file not found: {path}")

    with open(path) as f:
        mlir_str = f.read()

    mlir_module = mlir_ir.Module.parse(mlir_str, context=context)
    logger.debug(f"Loaded MLIR from: {path}")

    return mlir_module


def save_fx_as_mlir(
    gm: torch.fx.GraphModule,
    path: str | Path,
    output_type: OutputType = OutputType.RAW,
    verbose: bool = False,
    enable_ir_printing: bool = False,
) -> Path:
    """
    Convert FX GraphModule to MLIR and save to file

    Convenience function that combines fx_to_mlir_string and save_mlir.

    Args:
        gm: FX GraphModule to convert and save
        path: Output path (e.g., .raw.mlir, .stablehlo.mlir)
        output_type: MLIR output type (RAW, STABLEHLO, etc.)
        verbose: Enable verbose output
        enable_ir_printing: Enable IR printing during conversion

    Returns:
        Path: Path to saved file
    """
    mlir_str = fx_to_mlir_string(gm, output_type, verbose, enable_ir_printing)
    return save_mlir(mlir_str, path)


# ============================================================================
# Combined I/O for debugging
# ============================================================================


def save_fx_graph_all_formats(
    gm: torch.fx.GraphModule,
    base_path: str | Path,
    save_txt: bool = True,
    save_raw_mlir: bool = True,
    save_stablehlo_mlir: bool = False,
) -> dict:
    """
    Save FX GraphModule in multiple formats for debugging

    Args:
        gm: FX GraphModule to save
        base_path: Base path (without extension)
        save_txt: Save as .fx.txt (human-readable)
        save_raw_mlir: Save as .raw.mlir (RAW Torch dialect)
        save_stablehlo_mlir: Save as .stablehlo.mlir (StableHLO)

    Returns:
        dict: Mapping of format name to saved file path
    """
    base_path = Path(base_path)
    saved_files = {}

    if save_txt:
        txt_path = base_path.with_suffix(".fx.txt")
        saved_files["txt"] = save_fx_graph_txt(gm, txt_path)

    if save_raw_mlir:
        raw_path = base_path.with_suffix(".raw.mlir")
        saved_files["raw_mlir"] = save_fx_as_mlir(gm, raw_path, OutputType.RAW)

    if save_stablehlo_mlir:
        stablehlo_path = base_path.with_suffix(".stablehlo.mlir")
        saved_files["stablehlo_mlir"] = save_fx_as_mlir(gm, stablehlo_path, OutputType.STABLEHLO)

    logger.debug(f"Saved FX graph in {len(saved_files)} formats")

    return saved_files


# ============================================================================
# Artifacts
# ============================================================================
def cleanup_compilation_artifacts(temp_dir: Path, preserve: bool = True):
    """
    Clean up temporary compilation artifacts

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
