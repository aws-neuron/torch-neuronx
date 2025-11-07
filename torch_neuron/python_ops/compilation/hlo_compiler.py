"""HLO to NEFF compilation utilities."""

import logging

from torch_neuron.kernels.compiler_config import CompilerConfig
from torch_neuron.kernels.compiler_subprocess import CompilerSubprocess

logger = logging.getLogger(__name__)


class HloCompiler:
    """Handles compilation of HLO to NEFF."""

    def __init__(self, compiler_config: CompilerConfig | None = None):
        """Initialize the HLO compiler.

        Args:
            compiler_config: Compiler configuration
        """
        self.compiler_config = compiler_config or CompilerConfig()

    def compile_to_neff(self, hlo_protobuf: bytes) -> bytes:
        """Compile HLO protobuf to NEFF.

        Args:
            hlo_protobuf: HLO protobuf bytes

        Returns:
            Compiled NEFF bytes
        """
        # Get logical neuron cores
        lnc = self.compiler_config.lnc
        logger.debug(f"Compiling HLO to NEFF with {lnc} logical cores")

        # Use subprocess handler for compilation
        neff_bytes = CompilerSubprocess.compile_hlo_protobuf_to_neff(
            hlo_protobuf, self.compiler_config, lnc
        )

        logger.debug(f"Successfully compiled to NEFF ({len(neff_bytes)} bytes)")
        return neff_bytes

    def update_config(self, compiler_config: CompilerConfig) -> None:
        """Update the compiler configuration.

        Args:
            compiler_config: New compiler configuration
        """
        self.compiler_config = compiler_config
        logger.debug("Updated compiler configuration")
