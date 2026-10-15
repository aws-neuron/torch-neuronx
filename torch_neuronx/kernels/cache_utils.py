"""Cache configuration utilities."""

import hashlib
import logging
import os
import pickle
import secrets
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from torch_neuronx.utils import get_rank

logger = logging.getLogger(__name__)


def get_artifacts_directory() -> Path:
    """Return the directory used for compilation artifacts.

    Returns:
        Path: Configured artifact directory or ``/tmp/neuron_backend``.

    Note:
        Use ``TORCH_NEURONX_DEBUG_DIR`` when set; otherwise use
        ``/tmp/neuron_backend``.
    """
    debug_dir = os.getenv("TORCH_NEURONX_DEBUG_DIR")
    path = Path(debug_dir if debug_dir else "/tmp/neuron_backend")
    return path


def get_nfs_cache_dir() -> str:
    """Return the shared NEFF cache directory.

    Returns:
        str: Configured shared cache directory or ``/tmp/neff_cache``.

    Note:
        Use ``TORCH_NEURONX_NEFF_CACHE_DIR`` when set; otherwise use
        ``/tmp/neff_cache``.
    """
    return os.getenv("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache")


def get_local_cache_dir() -> str:
    """Return the local NEFF cache directory.

    Returns:
        str: Configured local cache directory or ``/tmp/local_cache``.

    Note:
        Use ``TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR`` when set; otherwise use
        ``/tmp/local_cache``.
    """
    return os.getenv("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "/tmp/local_cache")


def is_caching_disabled() -> bool:
    """Report whether NEFF caching is disabled.

    Returns:
        bool: ``True`` when ``TORCH_NEURONX_NEFF_DISABLE_CACHE`` is ``true``,
        ``1``, or ``yes``, ignoring case.
    """
    return os.getenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "").lower() in ("true", "1", "yes")


def get_lock_timeout() -> int:
    """Return the NEFF cache lock timeout in seconds.

    Returns:
        int: ``TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT``, or ``1200`` when unset.

    Raises:
        ValueError: If the configured timeout is not an integer.
    """
    return int(os.getenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "1200"))


def is_preserve_artifacts_enabled() -> bool:
    """Report whether compilation artifacts should be preserved.

    Returns:
        bool: ``True`` when ``TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS`` is
        ``true``, ``1``, or ``yes``, ignoring case.
    """
    return os.getenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "").lower() in (
        "true",
        "1",
        "yes",
    )


def atomic_write_bytes(file_path: Path, data: bytes) -> None:
    """Write bytes atomically by renaming a temporary file.

    Args:
        file_path (Path): Destination file path.
        data (bytes): Bytes to write.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in same directory
    with tempfile.NamedTemporaryFile(dir=file_path.parent, delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(data)
        tmp_path = Path(tmp_file.name)

    # Atomic rename
    tmp_path.rename(file_path)


class CompilationJob:
    """Manage artifact paths and lifecycle for one compilation job.

    Args:
        op_name (str): Operation name included in the unique job identifier.

    Note:
        Each job writes artifacts beneath a unique directory in the base artifact
        directory. FX graphs, transformed graphs, raw Torch IR, StableHLO, and
        Torch-MLIR errors use separate subdirectories. Neuron compiler work uses a
        hash-named directory under ``neuronx-cc``. The preservation setting controls
        whether these artifacts are removed during cleanup.
    """

    def __init__(self, op_name: str):
        self.op_name = op_name
        self.rank = get_rank()
        self.timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.job_id = self._generate_job_id()
        self.preserve = is_preserve_artifacts_enabled()
        self._base_dir = get_artifacts_directory()
        self._job_dir = self._base_dir / "torch_neuronx" / f"rank{self.rank}" / self.job_id
        self._pass_counter = 0
        working_dir_hash = hashlib.sha256(self.job_id.encode(errors="ignore")).hexdigest()
        self._neuronx_cc_directory = self._base_dir / "neuronx-cc" / working_dir_hash

    def _generate_job_id(self) -> str:
        """Generate a unique kernel ID for compilation tracking.

        The kernel ID format is: {timestamp}_{op_name}_{hash}
        where:
        - timestamp: Local timezone timestamp in format YYYYMMDD_HHMMSS_microseconds
        - op_name: Operation name truncated/padded to 16 chars, sanitized
        - hash: 16 character random hex string

        Sanitization: Replaces | and whitespace characters with _ for filesystem safety.
        """
        # Sanitize op_name: replace | and whitespace with _
        sanitized_op_name = (
            self.op_name.replace("|", "_").replace(" ", "_").replace("\t", "_").replace("\n", "_")
        )

        # Normalize op_name to 16 chars
        if len(sanitized_op_name) > 16:
            normalized_op_name = sanitized_op_name[:16]
        else:
            normalized_op_name = sanitized_op_name.ljust(16, "_")

        # Generate 16-char random hex hash
        random_hash = secrets.token_hex(8)

        return f"{self.timestamp}_{normalized_op_name}_{random_hash}"

    def _make_path(self, subdir: str, model_name: str, suffix: str) -> Path:
        """Create an artifact path, ensuring parent directory exists."""
        directory = self._job_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{model_name.lower()}_{self.timestamp}{suffix}"
        return directory / filename

    def save_fx_graph(self, model_name: str, gm) -> Path:
        """Save an FX graph as human-readable text.

        Args:
            model_name (str): Model name used in the artifact filename.
            gm (torch.fx.GraphModule): FX graph module to save.

        Returns:
            Path: Path to the saved ``fx_graphs`` artifact.
        """
        path = self._make_path("fx_graphs", model_name, ".fx.txt")
        with open(path, "w") as f:
            f.write(gm.print_readable(print_output=False))
        logger.debug(f"Saved FX graph to {path}")
        return path

    def save_pass_fx(self, model_name: str, gm, pass_name: str) -> Path:
        """Save an FX graph produced by an individual transformation pass.

        Args:
            model_name (str): Model name used in the artifact filename.
            gm (torch.fx.GraphModule): FX graph module to save.
            pass_name (str): Pass name included in the artifact filename.

        Returns:
            Path: Path to the indexed artifact under ``transformed_fx``.
        """
        pass_index = self._pass_counter
        self._pass_counter += 1
        suffix = f".pass_{pass_index}_{pass_name}.fx.txt"
        path = self._make_path("transformed_fx", model_name, suffix)
        with open(path, "w") as f:
            f.write(gm.print_readable(print_output=False))
        logger.debug(f"Saved FX graph after pass {pass_index} ({pass_name}) to {path}")
        return path

    def save_transformed_fx(self, model_name: str, gm) -> Path:
        """Save a transformed FX graph as human-readable text.

        Args:
            model_name (str): Model name used in the artifact filename.
            gm (torch.fx.GraphModule): Transformed FX graph module to save.

        Returns:
            Path: Path to the saved ``transformed_fx`` artifact.
        """
        path = self._make_path("transformed_fx", model_name, ".fx.txt")
        with open(path, "w") as f:
            f.write(gm.print_readable(print_output=False))
        logger.debug(f"Saved transformed FX graph to {path}")
        return path

    def save_raw_torch(self, model_name: str, mlir_module) -> Path:
        """Save raw Torch IR as text.

        Args:
            model_name (str): Model name used in the artifact filename.
            mlir_module (object): MLIR module whose operation is serialized.

        Returns:
            Path: Path to the saved ``raw`` Torch MLIR artifact.
        """
        path = self._make_path("raw", model_name, ".torch_raw.mlir")
        with open(path, "w") as f:
            f.write(mlir_module.operation.get_asm(enable_debug_info=True))
        logger.debug(f"Saved RAW Torch module to {path}")
        return path

    def save_stablehlo(self, model_name: str, mlir_module) -> Path:
        """Save StableHLO IR as text.

        Args:
            model_name (str): Model name used in the artifact filename.
            mlir_module (object): MLIR module whose operation is serialized.

        Returns:
            Path: Path to the saved ``stablehlo`` artifact.
        """
        path = self._make_path("stablehlo", model_name, ".stablehlo.mlir")
        with open(path, "w") as f:
            f.write(mlir_module.operation.get_asm(enable_debug_info=True))
        logger.debug(f"Saved StableHLO to {path}")
        return path

    def get_err_mlir_path(self, model_name: str) -> Path:
        """Create a path for a Torch-MLIR error artifact.

        Args:
            model_name (str): Model name used in the artifact filename.

        Returns:
            Path: Path under the job's ``torch_mlir_error`` directory.
        """
        return self._make_path("torch_mlir_error", model_name, ".mlir")

    def get_neuronx_cc_workdir(self) -> Path:
        """Return the Neuron compiler working directory.

        Returns:
            Path: Hash-named working directory under ``neuronx-cc``.
        """
        return self._neuronx_cc_directory

    def get_dump_dir(self) -> str:
        """Return this job's directory as a string.

        Returns:
            str: Compilation job directory.
        """
        return str(self._job_dir)

    def cleanup(self):
        """Remove compilation artifact directories unless preservation is enabled.

        Note:
            Remove both the job directory and Neuron compiler working directory when
            they exist. Log cleanup failures instead of raising them.
        """
        try:
            if self.preserve:
                logger.debug(
                    f"Preserving compilation artifacts: job={self._job_dir} "
                    f"neuronx-cc={self._neuronx_cc_directory}"
                )
                return
            if self._job_dir.exists():
                logger.debug(f"Cleaning up job directory: {self._job_dir}")
                shutil.rmtree(self._job_dir)
            if self._neuronx_cc_directory.exists():
                logger.debug(f"Cleaning up neuronx-cc directory: {self._neuronx_cc_directory}")
                shutil.rmtree(self._neuronx_cc_directory)

        except Exception as e:
            logger.warning(f"Failed to clean up {self._job_dir}: {e}")


@contextmanager
def compilation_job(op_name: str):
    """Create a compilation job that is cleaned up on context exit.

    Args:
        op_name (str): Operation name for the compilation job.

    Yields:
        CompilationJob: Job whose cleanup runs even when the context raises.
    """
    job = CompilationJob(op_name)
    try:
        yield job
    finally:
        job.cleanup()


@dataclass
class TensorDumpPolicy:
    """Configure tensor-dump directions and rank filters.

    Args:
        dump_inputs (bool, optional): Whether to dump inputs. Defaults to ``False``.
        dump_outputs (bool, optional): Whether to dump outputs. Defaults to ``False``.
        input_ranks (set[int], optional): Input ranks to dump. Defaults to an empty set.
        output_ranks (set[int], optional): Output ranks to dump. Defaults to an empty set.
        input_all_ranks (bool, optional): Whether to dump inputs on every rank.
            Defaults to ``False``.
        output_all_ranks (bool, optional): Whether to dump outputs on every rank.
            Defaults to ``False``.
    """

    dump_inputs: bool = False
    dump_outputs: bool = False
    input_ranks: set[int] = field(default_factory=set)
    output_ranks: set[int] = field(default_factory=set)
    input_all_ranks: bool = False
    output_all_ranks: bool = False

    def should_dump_inputs(self, rank: int) -> bool:
        """Check whether inputs should be dumped for a rank.

        Args:
            rank (int): Rank to check.

        Returns:
            bool: ``True`` when input dumping includes the rank.
        """
        if not self.dump_inputs:
            return False
        return self.input_all_ranks or rank in self.input_ranks

    def should_dump_outputs(self, rank: int) -> bool:
        """Check whether outputs should be dumped for a rank.

        Args:
            rank (int): Rank to check.

        Returns:
            bool: ``True`` when output dumping includes the rank.
        """
        if not self.dump_outputs:
            return False
        return self.output_all_ranks or rank in self.output_ranks


def _parse_tensor_dump_policy() -> TensorDumpPolicy:
    """Parse TORCH_NEURONX_DYNAMO_TENSOR_DUMPS into a TensorDumpPolicy.

    The environment variable holds semicolon-delimited ``key=value`` pairs.
    Keys are ``inputs`` or ``outputs``. Values are ``all`` or a
    comma-separated rank list.
    """
    env_value = os.environ.get("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "")

    if not env_value.strip():
        return TensorDumpPolicy()

    dump_inputs = False
    dump_outputs = False
    input_ranks: set[int] = set()
    output_ranks: set[int] = set()
    input_all_ranks = False
    output_all_ranks = False

    for part in env_value.split(";"):
        part = part.strip()
        if not part:
            continue

        if "=" not in part:
            logger.warning(f"Invalid tensor dump config entry (missing '='): '{part}'")
            continue

        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "inputs":
            dump_inputs = True
            if value.lower() == "all":
                input_all_ranks = True
            else:
                input_ranks = _parse_rank_list(value)
        elif key == "outputs":
            dump_outputs = True
            if value.lower() == "all":
                output_all_ranks = True
            else:
                output_ranks = _parse_rank_list(value)
        else:
            logger.warning(f"Unknown tensor dump config key: '{key}'")

    return TensorDumpPolicy(
        dump_inputs=dump_inputs,
        dump_outputs=dump_outputs,
        input_ranks=input_ranks,
        output_ranks=output_ranks,
        input_all_ranks=input_all_ranks,
        output_all_ranks=output_all_ranks,
    )


def _parse_rank_list(value: str) -> set[int]:
    """Parse a comma-separated list of rank integers."""
    ranks = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            try:
                ranks.add(int(item))
            except ValueError:
                logger.warning(f"Invalid rank value in tensor dump config: '{item}'")
    return ranks


class TensorDumper:
    """Write input and output tensor dumps for one executor.

    Args:
        dump_dir (str): Root directory for input and output dump subdirectories.

    Note:
        The policy and current rank are read on every dump call.
    """

    def __init__(self, dump_dir: str):
        self._dump_dir = Path(dump_dir)

    def dump_tensors(self, tensors: list, execution_count: int, is_input: bool) -> None:
        """Dump selected tensors as pickled NumPy arrays.

        Args:
            tensors (list[torch.Tensor]): Tensors that support transfer to CPU and
                conversion to NumPy.
            execution_count (int): Invocation count included in the filename.
            is_input (bool): Whether to apply input policy and naming instead of output
                policy and naming.

        Note:
            Writes ``inputs/inp-<rank>-<count>.p`` or
            ``outputs/out-<rank>-<count>.p`` beneath the dump root. Each file contains
            keys such as ``input0`` or ``output0``. Converts bfloat16 tensors to float32
            and logs conversion or write failures without raising.
        """
        policy = _parse_tensor_dump_policy()
        rank = get_rank()
        if is_input and not policy.should_dump_inputs(rank):
            return
        if not is_input and not policy.should_dump_outputs(rank):
            return

        prefix = "inp" if is_input else "out"
        key_prefix = "input" if is_input else "output"
        subdir = "inputs" if is_input else "outputs"
        rank_3digit = f"{rank:03d}"
        filename = f"{prefix}-{rank_3digit}-{execution_count}.p"
        filepath = self._dump_dir / subdir / filename

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            tensor_dict = {}
            for i, tensor in enumerate(tensors):
                t = tensor.detach().cpu()
                if t.dtype == torch.bfloat16:
                    t = t.float()
                tensor_dict[f"{key_prefix}{i}"] = t.numpy()

            with open(filepath, "wb") as f:
                pickle.dump(tensor_dict, f)

            logger.debug(f"Dumped {len(tensors)} {key_prefix} tensors to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to dump {key_prefix} tensors to {filepath}: {e}")
