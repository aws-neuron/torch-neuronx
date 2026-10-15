"""
I/O utilities for FX graphs and MLIR modules

This module provides utilities for saving and loading FX graphs and MLIR modules
in various formats for debugging and analysis purposes.
"""

import logging
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
    """Save an FX graph module as human-readable text.

    Args:
        gm (torch.fx.GraphModule): Graph module to render with
            ``print_readable``.
        path (str | Path): Destination file path.

    Returns:
        Path: Destination path after writing the graph text.
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
    """Convert an FX graph module to an MLIR string.

    Args:
        gm (torch.fx.GraphModule): Graph module to import.
        output_type (OutputType, optional): MLIR dialect to produce. Defaults to
            ``OutputType.RAW``.
        verbose (bool, optional): Enable verbose importer output. Defaults to ``False``.
        enable_ir_printing (bool, optional): Enable importer IR printing. Defaults to
            ``False``.

    Returns:
        str: Text representation of the imported MLIR module.
    """
    logger.debug(f"Converting FX graph to MLIR ({output_type})...")

    mlir_module = fx.stateless_fx_import(
        gm, output_type=output_type, verbose=verbose, enable_ir_printing=enable_ir_printing
    )

    mlir_str = str(mlir_module)
    logger.debug(f"Converted FX graph to MLIR ({output_type})")

    return mlir_str


def save_mlir(mlir_str: str, path: str | Path) -> Path:
    """Save an MLIR string to a file.

    Args:
        mlir_str (str): MLIR text to write.
        path (str | Path): Destination file path.

    Returns:
        Path: Destination path after writing the MLIR text.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(mlir_str)

    logger.debug(f"Saved MLIR to: {path}")

    return path


def load_mlir(path: str | Path, context=None) -> mlir_ir.Module:
    """Load an MLIR module from a file.

    Args:
        path (str | Path): Source MLIR file path.
        context (mlir_ir.Context, optional): Context passed to the parser.
            Defaults to ``None``.

    Returns:
        mlir_ir.Module: Parsed MLIR module.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
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
    """Convert an FX graph module to MLIR and save it.

    Args:
        gm (torch.fx.GraphModule): Graph module to import.
        path (str | Path): Destination MLIR file path.
        output_type (OutputType, optional): MLIR dialect to produce. Defaults to
            ``OutputType.RAW``.
        verbose (bool, optional): Enable verbose importer output. Defaults to ``False``.
        enable_ir_printing (bool, optional): Enable importer IR printing. Defaults to
            ``False``.

    Returns:
        Path: Destination path after writing the converted MLIR text.
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
    """Save an FX graph module in selected debugging formats.

    Args:
        gm (torch.fx.GraphModule): Graph module to save.
        base_path (str | Path): Base path whose suffix is replaced per format.
        save_txt (bool, optional): Save human-readable ``.fx.txt`` output. Defaults to
            ``True``.
        save_raw_mlir (bool, optional): Save RAW Torch dialect MLIR. Defaults to ``True``.
        save_stablehlo_mlir (bool, optional): Save StableHLO MLIR. Defaults to ``False``.

    Returns:
        dict[str, Path]: Saved paths keyed by ``"txt"``, ``"raw_mlir"``, and/or
            ``"stablehlo_mlir"`` for the enabled formats.
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
