#!/usr/bin/env python3
"""
Neuronx CC Wrapper - Command-line interface for cached compilation

This module provides a command-line wrapper around CompilerSubprocess.
Called as a subprocess, replacing direct neuronx-cc calls
with cached compilation.

Call flow:
C++ → Python subprocess (neuronx_cc_wrapper) → CompilerSubprocess (get cached/compile) → neuronx-cc

"""

import argparse
import logging
import os
import shlex
import sys
from dataclasses import dataclass

from .compiler_config import CompilerConfig
from .compiler_subprocess import CompilerSubprocess, NEFFCompilationError

logger = logging.getLogger(__name__)


@dataclass
class ParsedArgs:
    """Store parsed arguments for the ``neuronx-cc`` wrapper.

    Args:
        config (CompilerConfig): Parsed compiler configuration.
        lnc_override (str | None): Logical NeuronCore override, if parsed.
        input_file (str | None): Input HLO or MLIR file path, if parsed.
        output_file (str | None): Output NEFF file path, if parsed.
        workdir (str | None): Compiler working directory, if parsed.
    """

    config: CompilerConfig
    lnc_override: str | None
    input_file: str | None
    output_file: str | None
    workdir: str | None


class NeuronxCCWrapper:
    """Wrap cached compilation in a ``neuronx-cc`` command-line interface."""

    def __init__(self):
        """Initialize the wrapper."""
        self.compiler = CompilerSubprocess()

    def compile_with_caching(self, args: list[str]) -> int:
        """Compile an input file through the NEFF cache.

        Args:
            args (list[str]): ``neuronx-cc`` command-line arguments.

        Returns:
            int: ``0`` on success and ``1`` for missing parsed paths, input-reading
            errors, or handled cache and compilation failures.
        """
        try:
            parsed = self._parse_neuronx_cc_args(args)

            if not parsed.input_file:
                logger.error("Could not parse input file from arguments")
                return 1
            if not parsed.output_file:
                logger.error("Could not parse output file from arguments")
                return 1
            if not parsed.workdir:
                logger.error("Could not parse working directory from arguments")
                return 1

            # Read input file
            try:
                with open(parsed.input_file, "rb") as f:
                    hlo_protobuf = f.read()
            except Exception as e:
                logger.error(f"Error reading input file {parsed.input_file}: {e}")
                return 1

            # Determine IR type based on file extension
            ir_type = "StableHLO"

            # Get cached neff or compile
            self.compiler.get_or_compile(
                hlo_protobuf=hlo_protobuf,
                config=parsed.config,
                lnc_override=parsed.lnc_override,
                ir_type=ir_type,
                output_file=parsed.output_file,
            )

            logger.info(f"Successfully compiled {parsed.input_file} -> {parsed.output_file}")
            return 0

        except NEFFCompilationError as e:
            logger.error(f"NEFF compilation failed: {e}")
            return 1
        except Exception as e:
            logger.error(f"Error occurred during compilation/cached neff retrieval: {e}")
            return 1

    def _parse_neuronx_cc_args(self, args: list[str]) -> ParsedArgs:
        """Parse neuronx-cc arguments using argparse.

        Args:
            args: List of neuronx-cc command line arguments

        Returns:
            ParsedArgs containing config and runtime parameters
        """
        # Create parser for neuronx-cc arguments
        parser = argparse.ArgumentParser(add_help=False)

        parser.add_argument("command", nargs="?", help="compile command")
        parser.add_argument("input", nargs="?", help="Input HLO/MLIR file")
        parser.add_argument("--output", type=str, help="Output NEFF file")
        parser.add_argument("--framework", type=str, default="XLA", help="Framework")
        parser.add_argument("--target", type=str, help="Target platform")
        parser.add_argument("--model-type", type=str, default="transformer", help="Model type")
        parser.add_argument("--lnc", type=str, help="Logical neuron cores")
        parser.add_argument("--auto-cast", type=str, default="none", help="Auto-casting")
        parser.add_argument("--workdir", type=str, help="Working Directory")

        # Handle optimization flags (-O1, -O2, etc.)
        parser.add_argument("-O1", action="store_const", const="-O1", dest="optimization")
        parser.add_argument("-O2", action="store_const", const="-O2", dest="optimization")
        parser.add_argument("-O3", action="store_const", const="-O3", dest="optimization")

        parsed_args, extra_args = parser.parse_known_args(args)

        # Prepend extra compiler args from NEURON_CC_FLAGS environment variable
        neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "").strip()
        if neuron_cc_flags:
            try:
                extra_flags = shlex.split(neuron_cc_flags)
                extra_args = extra_flags + extra_args
                logger.debug(f"Prepended NEURON_CC_FLAGS: {extra_flags}")
            except ValueError as e:
                logger.warning(f"Failed to parse NEURON_CC_FLAGS '{neuron_cc_flags}': {e}")

        # Create CompilerConfig from parsed arguments
        config_kwargs = {}

        if parsed_args.framework:
            config_kwargs["framework"] = parsed_args.framework
        if parsed_args.target:
            config_kwargs["target"] = parsed_args.target
        if parsed_args.model_type:
            config_kwargs["model_type"] = parsed_args.model_type
        if parsed_args.optimization:
            config_kwargs["optimization_level"] = parsed_args.optimization
        if parsed_args.auto_cast:
            config_kwargs["auto_cast"] = parsed_args.auto_cast

        # Add any extra flags
        if extra_args:
            config_kwargs["extra_flags"] = extra_args

        config = CompilerConfig(**config_kwargs)

        return ParsedArgs(
            config=config,
            lnc_override=parsed_args.lnc,
            input_file=parsed_args.input,
            output_file=parsed_args.output,
            workdir=parsed_args.workdir,
        )


def main(args=None):
    """Run the cached ``neuronx-cc`` command-line wrapper.

    Args:
        args (iterable[str], optional): Arguments to parse. Defaults to
            ``sys.argv[1:]``.

    Returns:
        int: Status from ``compile_with_caching``: ``0`` on success or ``1`` for a
            handled failure.
    """
    args = sys.argv[1:] if args is None else list(args)

    try:
        wrapper = NeuronxCCWrapper()
        return wrapper.compile_with_caching(args)

    except Exception as e:
        logger.error(f"Error in neuronx_cc_wrapper: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
