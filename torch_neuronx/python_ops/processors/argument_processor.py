"""Argument processing utilities for Neuron operations."""

import logging
import os

import torch

from torch_neuronx.dtype_utils import torch_dtype_to_scalar_type
from torch_neuronx.utils import (
    convert_for_neuron,
    is_sync_mode_enabled,
)

logger = logging.getLogger(__name__)


class ArgumentProcessor:
    """Process arguments for Neuron operations.

    Args:
        static_argnames (tuple[str, ...], optional): Keyword argument names to keep
            static. Defaults to ``None``.
    """

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
        """Return the configured static positional argument indices.

        Returns:
            tuple[int, ...]: Static positional argument indices.

        Raises:
            RuntimeError: If the indices have not been configured.
        """
        if self._static_argnums is None:
            raise RuntimeError("static_argnums must be set before use")
        return self._static_argnums

    @static_argnums.setter
    def static_argnums(self, value: tuple[int, ...]) -> None:
        """Set the static positional argument indices.

        Args:
            value (tuple[int, ...]): Indices to store.
        """
        self._static_argnums = value

    def preprocess_kwargs(self, kwargs: dict) -> dict:
        """Convert dynamic numeric keyword arguments to Neuron tensors.

        Args:
            kwargs (dict): Keyword arguments to process.

        Returns:
            dict: Keyword arguments with exact ``int`` and ``float`` values converted
            to ``int32`` and ``float32`` tensors on the Neuron device, respectively.
        """
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

    def preprocess_inputs(
        self, inputs: tuple, scalar_to_tensor: bool = False, convert_int64: bool = True
    ) -> tuple:
        """Prepare positional inputs for Neuron compatibility.

        Args:
            inputs (tuple): Input tensors, scalars, dtypes, and nested structures.
            scalar_to_tensor (bool, optional): Convert ``int`` and ``float`` scalars
                to tensors. Defaults to ``False``.
            convert_int64 (bool, optional): Apply enabled ``int64`` compatibility
                conversion. Defaults to ``True``; packed mode retains logical ``int64``.

        Returns:
            tuple: Inputs converted for Neuron lowering while static inputs remain unchanged.

        Raises:
            RuntimeError: If static positional indices have not been configured.
            OverflowError: If packed-mode checking finds ``int64`` values outside ``int32``.

        Note:
            Nested sequences become tuples, and ``torch.dtype`` values become scalar-type
            integers. Scalar conversions use the first top-level non-scalar tensor's device,
            or the current Neuron device when no such tensor exists.
        """
        # Prologues are enabled by default, but disabled if:
        # 1. TORCH_NEURONX_ENABLE_PROLOGUE env var is "0" or "false"
        # 2. OR sync mode is enabled
        prologue_disabled = (
            os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false")
            or is_sync_mode_enabled()
        )

        # Determine target device from first non-scalar tensor
        target_device = None
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and inp.ndim > 0:
                target_device = inp.device
                break

        # Fallback device for scalar-to-tensor conversion when there are no
        # tensor inputs (creation ops like scalar_tensor).  Uses the current
        # process's assigned device so distributed ranks don't all target
        # neuron:0.
        if target_device is None and scalar_to_tensor:
            import torch_neuronx

            scalar_device = torch.device("neuron", torch_neuronx.current_device())
        else:
            scalar_device = None

        def preprocess_value(inp):
            """Prepare one input value for Neuron compatibility.

            Args:
                inp (object): Input value to process.

            Returns:
                object: Converted tensor, tuple, scalar-type integer, or original value.
            """
            if isinstance(inp, torch.Tensor):
                # Move CPU scalar tensors to target device (matches CUDA semantics)
                if target_device and inp.device.type == "cpu" and inp.ndim == 0:
                    inp = inp.to(target_device)
                # Make tensor contiguous if prologue is disabled and tensor is not contiguous
                if prologue_disabled and not inp.is_contiguous():
                    inp = inp.contiguous()
                return convert_for_neuron(inp, convert_int64=convert_int64)
            elif isinstance(inp, (list | tuple)):
                return tuple(preprocess_value(item) for item in inp)
            elif isinstance(inp, torch.dtype):
                # Map torch.dtype to ScalarType enum values as defined in
                # c10/core/ScalarType.h for MLIR lowering
                return ArgumentProcessor._get_scalar_type_value(inp)
            elif scalar_to_tensor and isinstance(inp, int):
                return torch.tensor(inp, dtype=torch.int32, device=target_device or scalar_device)
            elif scalar_to_tensor and isinstance(inp, float):
                return torch.tensor(inp, dtype=torch.float32, device=target_device or scalar_device)
            else:
                # Keep scalars as-is
                return inp

        return tuple(
            inp if i in self.static_argnums else preprocess_value(inp)
            for i, inp in enumerate(inputs)
        )

    def generate_input_signature(self, inputs: tuple) -> tuple:
        """Build a cache signature from positional inputs.

        Args:
            inputs (tuple): Input tensors and scalar values.

        Returns:
            tuple: Shape-dtype pairs for top-level tensors, values for static
            non-tensors, and type names for other runtime values.

        Raises:
            RuntimeError: If static indices are unset when a non-tensor is classified.
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
        """Build the tensor input mapping for NEFF execution.

        Args:
            inputs (tuple): Preprocessed positional inputs.
            device (torch.device): Target execution device.

        Returns:
            dict[str, torch.Tensor]: Sequential names mapped to converted top-level
            tensors, one-level sequence tensor or numeric elements, and dynamic scalars.

        Raises:
            RuntimeError: If static indices are unset when a non-tensor is classified.
            OverflowError: If packed-mode checking finds ``int64`` values outside ``int32``.
        """

        def to_device(tensor: torch.Tensor) -> torch.Tensor:
            # Match device conversion semantics of CUDA, where input scalar (0-dimensional) CPU
            # tensors are automatically moved to the correct device
            """Move an eligible CPU scalar tensor to the execution device.

            Args:
                tensor (torch.Tensor): Tensor to inspect.

            Returns:
                torch.Tensor: Tensor on its original device, or ``device`` for a
                zero-dimensional CPU tensor.
            """
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
        """Select the execution device from inputs and keyword arguments.

        Args:
            inputs (tuple): Input tensors and scalar values.
            kwargs (dict, optional): Keyword arguments that may specify ``device``.
                Defaults to ``None``.

        Returns:
            torch.device: The first top-level tensor's device, an explicit nondefault
            device, or the process's current Neuron device.

        Note:
            A bare Neuron device or ``neuron:0`` keyword is remapped to the process's
            current Neuron device when no input tensor determines the device.
        """
        # Try to get device from an input tensor first — this is the most
        # reliable source because the tensor is already on the correct device.
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                return inp.device

        # No tensor inputs (creation ops like scalar_tensor, ones, empty, etc.)
        # In distributed settings each rank is restricted to its own device
        # slice via NEURON_RT_VISIBLE_CORES.  PyTorch's dispatch fills in the
        # default device (neuron:0) for creation-op kwargs, which is wrong for
        # ranks assigned to a different device.  Use current_device() to get
        # the process's actual assigned device.
        import torch_neuronx

        current_dev = torch.device("neuron", torch_neuronx.current_device())

        # If an explicit device kwarg is present, honour it — but remap the
        # bare default "neuron:0" to the current device so that decomposition-
        # generated ops (e.g. scalar_tensor from add_) land on the right core.
        if kwargs and kwargs.get("device") is not None:
            dev = torch.device(kwargs["device"])
            if dev.type == "neuron" and (dev.index is None or dev.index == 0):
                return current_dev
            return dev

        return current_dev

    def add_static_to_cache_key(self, base_key: str, kwargs: dict) -> str:
        """Append static keyword arguments to a cache key.

        Args:
            base_key (str): Base cache key.
            kwargs (dict): Keyword arguments containing possible static values.

        Returns:
            str: Cache key extended with configured static keyword values.
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
        val = torch_dtype_to_scalar_type(dtype)
        return val if val is not None else 6  # Default to Float (6)
