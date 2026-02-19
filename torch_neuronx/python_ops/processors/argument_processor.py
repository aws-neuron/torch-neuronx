"""Argument processing utilities for JAX operations."""

import logging
import os

import torch

from torch_neuronx.utils import convert_for_neuron, is_stablehlo_enabled, is_sync_mode_enabled

logger = logging.getLogger(__name__)


class ArgumentProcessor:
    """Processes and prepares arguments for JAX operations."""

    def __init__(
        self,
        static_argnames: tuple[str, ...] | None = None,
    ):
        """Initialize the processor.

        Args:
            static_argnames: Names of static keyword arguments
        """
        self._static_argnums = None
        self.static_argnames = static_argnames or ()

    @property
    def static_argnums(self) -> tuple[int, ...]:
        """Get static_argnums."""
        if self._static_argnums is None:
            raise RuntimeError("static_argnums must be set before use")
        return self._static_argnums

    @static_argnums.setter
    def static_argnums(self, value: tuple[int, ...]) -> None:
        """Set static_argnums."""
        self._static_argnums = value

    def preprocess_kwargs(self, kwargs: dict) -> dict:
        """Preprocess kwargs by converting non-static int/float to tensors."""
        if not kwargs:
            return kwargs

        processed_kwargs = {}
        for key, value in kwargs.items():
            if key not in self.static_argnames and (type(value) is int or type(value) is float):
                if type(value) is int:
                    processed_kwargs[key] = torch.tensor(value, dtype=torch.int32, device="neuron")
                else:
                    processed_kwargs[key] = torch.tensor(
                        value, dtype=torch.float32, device="neuron"
                    )
            else:
                processed_kwargs[key] = value
        return processed_kwargs

    def preprocess_inputs(self, inputs: tuple, scalar_to_tensor: bool = False) -> tuple:
        """Preprocess inputs for Neuron compatibility.

        Converts tensors, handles nested structures, maps torch.dtype to scalar type
        integers, and optionally converts scalar values to tensors.

        Args:
            inputs: Original input tensors, scalars, and nested structures
            scalar_to_tensor: If True, convert int/float scalars to tensors
                Defaults to False.

        Returns:
            Tuple of preprocessed inputs
        """

        # Prologues are enabled by default, but disabled if:
        # 1. StableHLO is not enabled (required for prologue)
        # 2. OR TORCH_NEURONX_ENABLE_PROLOGUE env var is "0" or "false"
        # 3. OR sync mode is enabled
        prologue_disabled = (
            not is_stablehlo_enabled()
            or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false")
            or is_sync_mode_enabled()
        )

        # Determine target device from first non-scalar tensor
        target_device = None
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and inp.ndim > 0:
                target_device = inp.device
                break

        def preprocess_value(inp):
            """preprocess a single input"""
            if isinstance(inp, torch.Tensor):
                # Move CPU scalar tensors to target device (matches CUDA semantics)
                if target_device and inp.device.type == "cpu" and inp.ndim == 0:
                    inp = inp.to(target_device)
                # Make tensor contiguous if prologue is disabled and tensor is not contiguous
                if prologue_disabled and not inp.is_contiguous():
                    inp = inp.contiguous()
                return convert_for_neuron(inp)
            elif isinstance(inp, (list | tuple)):
                return tuple(preprocess_value(item) for item in inp)
            elif isinstance(inp, torch.dtype):
                # Map torch.dtype to ScalarType enum values as defined in
                # c10/core/ScalarType.h for MLIR lowering
                return ArgumentProcessor._get_scalar_type_value(inp)
            elif scalar_to_tensor and isinstance(inp, int):
                return torch.tensor(inp, dtype=torch.int32, device="neuron")
            elif scalar_to_tensor and isinstance(inp, float):
                return torch.tensor(inp, dtype=torch.float32, device="neuron")
            else:
                # Keep scalars as-is
                return inp

        return tuple(
            inp if i in self.static_argnums else preprocess_value(inp)
            for i, inp in enumerate(inputs)
        )

    def generate_input_signature(self, inputs: tuple) -> tuple:
        """Generate a signature for inputs (used for caching).

        Args:
            inputs: Input tensors and scalars

        Returns:
            Tuple representing the input signature
        """
        signature_parts = []

        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                signature_parts.append((inp.shape, inp.dtype))
            elif i in self.static_argnums:
                # Include actual value for static args
                signature_parts.append(("static", inp))
            else:
                # Include type for runtime scalars
                signature_parts.append((type(inp).__name__,))

        return tuple(signature_parts)

    def prepare_execution_inputs(
        self, inputs: tuple, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Prepare input dictionary for NEFF execution.

        Args:
            inputs: Preprocessed inputs
            device: Target device

        Returns:
            Dictionary mapping input names to tensors
        """

        def to_device(tensor: torch.Tensor) -> torch.Tensor:
            # Match device conversion semantics of CUDA, where input scalar (0-dimensional) CPU
            # tensors are automatically moved to the correct device
            if tensor.device != device and tensor.device.type == "cpu" and tensor.ndim == 0:
                tensor = tensor.to(device)
            return tensor

        input_dict = {}
        tensor_idx = 0

        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                input_dict[f"input{tensor_idx}"] = convert_for_neuron(to_device(inp))
                tensor_idx += 1
            elif inp is not None and i not in self.static_argnums:
                if isinstance(inp, list | tuple):
                    # Handle tuple of tensors and mixed structures for ops like index_put, index
                    for val in inp:
                        if isinstance(val, torch.Tensor):
                            input_dict[f"input{tensor_idx}"] = convert_for_neuron(to_device(val))
                            tensor_idx += 1
                        elif isinstance(val, float):
                            scalar_tensor = torch.tensor(val, dtype=torch.float32, device=device)
                            input_dict[f"input{tensor_idx}"] = scalar_tensor
                            tensor_idx += 1
                        elif isinstance(val, int):
                            scalar_tensor = torch.tensor(val, dtype=torch.int32, device=device)
                            input_dict[f"input{tensor_idx}"] = scalar_tensor
                            tensor_idx += 1
                else:
                    # Convert non-static scalars to tensors
                    if isinstance(inp, float):
                        scalar_tensor = torch.tensor(inp, dtype=torch.float32, device=device)
                    else:
                        scalar_tensor = torch.tensor(inp, dtype=torch.int32, device=device)
                    input_dict[f"input{tensor_idx}"] = scalar_tensor
                    tensor_idx += 1
            # Static arguments are baked into compilation
        logger.debug(f"Prepared {len(input_dict)} inputs for execution")
        return input_dict

    def extract_device(self, inputs: tuple, kwargs: dict | None = None) -> torch.device:
        """Extract device from input tensors.

        Args:
            inputs: Input tensors and scalars
            kwargs: Keyword arguments (may contain 'device')

        Returns:
            Device from kwargs['device'], first tensor, or "neuron"
        """
        # Check for explicit device argument first
        if kwargs and kwargs.get("device") is not None:
            return torch.device(kwargs["device"])

        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                return inp.device
        return torch.device("neuron")

    def add_static_to_cache_key(self, base_key: str, kwargs: dict) -> str:
        """Add static keyword arguments to cache key.

        Args:
            base_key: Base cache key
            kwargs: Keyword arguments

        Returns:
            Extended cache key including static kwargs
        """
        if not self.static_argnames or not kwargs:
            return base_key

        static_parts = []
        for name in self.static_argnames:
            if name in kwargs:
                value = kwargs[name]
                static_parts.append(f"{name}_{value}")

        if static_parts:
            return f"{base_key}_{'_'.join(static_parts)}"

        return base_key

    @staticmethod
    def _get_scalar_type_value(dtype: torch.dtype) -> int:
        """Convert torch.dtype to ScalarType enum value.

        Maps torch.dtype to the corresponding integer values from c10::ScalarType enum
        as defined in c10/core/ScalarType.h. The order follows the
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS macro.

        Args:
            dtype: PyTorch dtype to convert

        Returns:
            Integer value corresponding to the ScalarType enum
        """
        dtype_to_scalar_type = {
            torch.uint8: 0,  # Byte
            torch.int8: 1,  # Char
            torch.int16: 2,  # Short
            torch.int32: 3,  # Int
            torch.int64: 4,  # Long
            torch.float16: 5,  # Half
            torch.float32: 6,  # Float
            torch.float64: 7,  # Double
            # torch.complex32: 8,   # ComplexHalf (not commonly used)
            torch.complex64: 9,  # ComplexFloat
            torch.complex128: 10,  # ComplexDouble
            torch.bool: 11,  # Bool
            # Quantized types: 12-14, 16-17
            torch.bfloat16: 15,  # BFloat16
            # Float8 types: 23-26
            # Additional types: 27-45
        }

        return dtype_to_scalar_type.get(dtype, 6)  # Default to Float (6)
