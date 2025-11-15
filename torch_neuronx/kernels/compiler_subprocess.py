"""Subprocess operations for HLO and NEFF compilation."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from .compiler_config import CompilerConfig


class CompilationError(Exception):
    """Base exception for compilation errors."""

    pass


class NEFFCompilationError(CompilationError):
    """Exception raised when NEFF compilation fails."""

    pass


class CompilerSubprocess:
    """Handles subprocess operations for compilation pipeline."""

    @staticmethod
    def find_neuronx_cc() -> str:
        """Find neuronx-cc compiler in PATH.

        Returns:
            Path to neuronx-cc executable

        Raises:
            RuntimeError: If neuronx-cc is not found
        """
        neuronx_cc = shutil.which("neuronx-cc")
        if not neuronx_cc:
            raise RuntimeError("neuronx-cc not found in PATH")
        return neuronx_cc

    @staticmethod
    def run_neuronx_cc(
        input_file: Path,
        output_file: Path,
        config: CompilerConfig,
        lnc_override: str | None = None,
        work_dir: str | None = None,
    ) -> None:
        """Compile HLO protobuf to NEFF.

        Args:
            input_file: Path to HLO protobuf file
            output_file: Path to output NEFF file
            config: Compiler configuration
            lnc_override: Optional override for logical neuron cores

        Raises:
            NEFFCompilationError: If compilation fails
        """
        neuronx_cc = CompilerSubprocess.find_neuronx_cc()
        cmd = [
            neuronx_cc,
            *config.get_neuronx_cc_args(str(input_file), str(output_file), lnc_override),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
        if result.returncode != 0:
            raise NEFFCompilationError(
                f"NEFF compilation failed with return code {result.returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {result.stderr}"
            )

    @staticmethod
    def compile_hlo_protobuf_to_neff(
        hlo_protobuf: bytes,
        config: CompilerConfig,
        lnc_override: str | None = None,
        ir_type: str = "XLA",
    ) -> bytes:
        """Compile HLO protobuf or MLIR directly to NEFF.

        Args:
            hlo_protobuf: HLO protobuf bytes or MLIR bytes
            config: Compiler configuration
            lnc_override: Optional override for logical neuron cores
            ir_type: Type of IR ("XLA" for HLO protobuf, "StableHLO" for MLIR)

        Returns:
            Compiled NEFF bytes

        Raises:
            NEFFCompilationError: If NEFF compilation fails
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Determine file format based on ir_type
            # "StableHLO" uses MLIR format, "XLA" uses HLO protobuf
            hlo_file = (
                tmpdir / "module.mlir" if ir_type == "StableHLO" else tmpdir / "module.hlo.pb"
            )

            hlo_file.write_bytes(hlo_protobuf)

            # Compile to NEFF
            neff_file = tmpdir / "module.neff"
            # Passing work_dir is important when we run multi-worker
            # Seeing a race condition where the artifacts get mangled
            # causing runtime errors. Passing the workdir keeps all
            # artifacts contained in their respective folder.
            CompilerSubprocess.run_neuronx_cc(
                hlo_file, neff_file, config, lnc_override, work_dir=tmpdir
            )

            # Read and return compiled NEFF
            return neff_file.read_bytes()
