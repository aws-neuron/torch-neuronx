"""Hilo frontend + XLAInferGoldens executor.

Verifies the neuronx-cc frontend (hilo) by:
1. Writing the StableHLO MLIR to a temp directory
2. Dumping inputs as .npy files
3. Invoking neuronx-cc with --pipeline HLOToTensorizer XLAInferGoldens
4. Reading back golden output .npy files as torch tensors
"""

import io
import logging
import os
import pickle
import subprocess
import tempfile

import numpy as np
import torch

from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO

logger = logging.getLogger(__name__)


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to numpy, handling bfloat16 specially.

    For bf16 we use a uint16 view to preserve the 2-byte-per-element buffer
    size that XLAInferGoldens expects.
    """
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().view(np.uint16)
    return t.numpy()


def _numpy_to_tensor(arr: np.ndarray, target_dtype: torch.dtype) -> torch.Tensor:
    """Convert a numpy array to a torch tensor with the target dtype."""
    if target_dtype == torch.bfloat16:
        raw = arr.view(np.uint16) if arr.dtype != np.uint16 else arr
        return torch.from_numpy(raw.copy()).view(torch.bfloat16)
    if target_dtype == torch.bool:
        return torch.from_numpy(arr.astype(np.bool_))
    return torch.from_numpy(arr.copy()).to(target_dtype)


def _get_target() -> str:
    """Get the neuronx-cc --target"""
    return "trn2"


class HiloInferGoldensExecutor:
    """Executor that runs stablehlo through neuronx-cc frontend + XLAInferGoldens.

    This tests that the neuronx-cc frontend correctly transforms
    the StableHLO produced by torch-neuronx and that XLA's reference
    interpreter produces correct golden outputs.
    """

    def __init__(self, stablehlo_mlir, io_spec: FunctionIO, cast_spec):
        self.stablehlo_mlir = stablehlo_mlir
        self.io_spec = io_spec
        self.cast_spec = cast_spec

    def __call__(self, inputs):
        if os.environ.get("TORCH_NEURONX_PRESERVE_HILO_WORKDIR") == "1":
            workdir = tempfile.mkdtemp(prefix="hilo_ig_")
            logger.debug(f"Preserving neuronx-cc workdir: {workdir}")
            return self._run(inputs, workdir)
        with tempfile.TemporaryDirectory(prefix="hilo_ig_") as workdir:
            return self._run(inputs, workdir)

    def _run(self, inputs, workdir: str):
        # 1. Write StableHLO MLIR to file
        mlir_path = os.path.join(workdir, "module.mlir")
        buf = io.StringIO()
        self.stablehlo_mlir.operation.print(file=buf, enable_debug_info=False)
        mlir_text = buf.getvalue()
        with open(mlir_path, "w") as f:
            f.write(mlir_text)
        logger.debug(f"Wrote StableHLO MLIR to {mlir_path}")

        # 2. Dump inputs as inp-000.p pickle
        inp_dict = {}
        for i, tensor in enumerate(inputs):
            if isinstance(tensor, torch.Tensor):  # noqa: SIM108
                arr = _tensor_to_numpy(tensor)
            else:
                arr = np.array(tensor)
            name = f"input{i}"
            inp_dict[name] = arr
        pickle_path = os.path.join(workdir, "inp-000.p")
        with open(pickle_path, "wb") as f:
            pickle.dump(inp_dict, f)
        logger.debug(f"Wrote {len(inputs)} input tensors to {workdir}")

        # 3. Build and run neuronx-cc command
        target = _get_target()
        cmd = [
            "neuronx-cc",
            "compile",
            mlir_path,
            "--framework",
            "XLA",
            "--target",
            target,
            "--pipeline",
            "HLOToTensorizer",
            "XLAInferGoldens",
        ]
        extra_flags = os.environ.get("NEURON_CC_FLAGS", "")
        if extra_flags:
            cmd.extend(extra_flags.split())
        logger.info(f"Running neuronx-cc: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"neuronx-cc timed out after 600s. Command: {' '.join(cmd)}") from e

        if result.returncode != 0:
            raise RuntimeError(
                f"neuronx-cc failed (exit {result.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
            )
        logger.debug("neuronx-cc completed successfully")

        # 4. Read golden outputs
        outputs = self._read_golden_outputs(workdir)
        return outputs

    def _read_golden_outputs(self, workdir: str) -> list:
        """Read golden outputs from the neuronx-cc working directory.

        neuronx-cc runs in a temp subdirectory. We need to find the gold-000.p
        or output npy files.
        """
        # neuronx-cc creates its own temp working directory inside workdir.
        # Look for gold-000.p recursively.
        gold_pickle = self._find_file(workdir, "gold-000.p")
        return self._read_from_pickle(gold_pickle)

    def _find_file(self, root: str, filename: str) -> str | None:
        """Recursively find a file by name."""
        for dirpath, _, filenames in os.walk(root):
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

    def _read_from_pickle(self, pickle_path: str) -> list:
        """Read outputs from gold-000.p pickle file."""
        with open(pickle_path, "rb") as f:
            gold_dict = pickle.load(f)
        logger.debug(f"Read golden outputs from {pickle_path}: {list(gold_dict.keys())}")

        # Gold dict has keys like 'output0', 'output1', etc.
        num_outputs = len(self.io_spec.outputs)
        outputs = []
        for i in range(num_outputs):
            key = f"output{i}"
            if key not in gold_dict:
                raise RuntimeError(
                    f"Expected output key '{key}' not found in gold-000.p. "
                    f"Available keys: {list(gold_dict.keys())}"
                )
            arr = gold_dict[key]
            target_dtype = self.io_spec.outputs[i].dtype
            tensor = _numpy_to_tensor(arr, target_dtype)

            # Apply cast_spec if needed
            if self.cast_spec and i < len(self.cast_spec):
                cast_dtype = self.cast_spec[i].dtype
                if cast_dtype != target_dtype:
                    tensor = tensor.to(cast_dtype)

            outputs.append(tensor)

        return outputs
