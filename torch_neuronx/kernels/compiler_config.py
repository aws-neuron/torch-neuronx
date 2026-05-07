"""Compiler configuration for XLA/NEFF compilation."""

from dataclasses import dataclass, field

from ..utils import get_logical_neuron_cores, get_platform_target


@dataclass
class CompilerConfig:
    """Configuration for neuronx-cc compiler.

    This class encapsulates all compiler settings to avoid hardcoding
    them throughout the codebase.
    """

    # Target hardware
    target: str = field(default_factory=get_platform_target)

    # Model type hint for optimization
    model_type: str = "transformer"

    # Optimization level
    optimization_level: str = "-O1"

    # Auto-casting behavior
    auto_cast: str = "none"

    # Framework
    framework: str = "XLA"

    # Logical Neuron Cores (LNC)
    lnc: str = field(default_factory=get_logical_neuron_cores)

    # Additional compiler flags
    extra_flags: list[str] = field(default_factory=list)

    def get_neuronx_cc_args(
        self,
        input_file: str,
        output_file: str,
        lnc_value: str | None = None,
    ) -> list[str]:
        """Generate neuronx-cc command line arguments.

        Args:
            input_file: Path to input HLO protobuf file
            output_file: Path to output NEFF file
            lnc_value: Override for logical neuron cores (uses self.lnc if not provided)

        Returns:
            List of command line arguments for neuronx-cc
        """
        args = [
            "compile",
            input_file,
            "--framework",
            self.framework,
            "--target",
            self.target,
            "--model-type",
            self.model_type,
            "--lnc",
            lnc_value or self.lnc,
            self.optimization_level,
            f"--auto-cast={self.auto_cast}",
            "--output",
            output_file,
        ]

        # Add any extra flags
        args.extend(self.extra_flags)

        return args
