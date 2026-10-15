"""Compiler configuration for XLA/NEFF compilation."""

from dataclasses import dataclass, field

from ..utils import get_logical_neuron_cores, get_platform_target


@dataclass
class CompilerConfig:
    """Configure a ``neuronx-cc`` invocation.

    Args:
        target (str, optional): Target platform. Defaults to the detected platform.
        model_type (str, optional): Model optimization hint. Defaults to ``transformer``.
        optimization_level (str, optional): Optimization flag. Defaults to ``-O1``.
        auto_cast (str, optional): Automatic casting mode. Defaults to ``none``.
        framework (str, optional): Input framework. Defaults to ``XLA``.
        lnc (str, optional): Logical NeuronCore count. Defaults to the detected value.
        extra_flags (list[str], optional): Additional compiler flags. Defaults to an
            empty list.

    Raises:
        RuntimeError: If the default target or logical NeuronCore count cannot be detected.
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
        """Build command-line arguments for ``neuronx-cc``.

        Args:
            input_file (str): Input HLO or MLIR file path.
            output_file (str): Output NEFF file path.
            lnc_value (str | None, optional): Logical NeuronCore override. A falsey value
                uses ``self.lnc``. Defaults to ``None``.

        Returns:
            list[str]: Compiler arguments beginning with the ``compile`` command.
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
