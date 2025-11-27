"""Argument processing utilities for JAX operations."""

import logging

import torch

from torch_neuronx.kernels.type_converter import TypeConverter

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

    def preprocess_inputs(self, inputs: tuple) -> tuple:
        """Preprocess inputs for Neuron compatibility.

        Args:
            inputs: Original input tensors and scalars

        Returns:
            Tuple of preprocessed inputs
        """
        converted = []

        for i, inp in enumerate(inputs):
            # Skip static arguments
            if i in self.static_argnums:
                converted.append(inp)
            elif isinstance(inp, torch.Tensor):
                # Ensure contiguous and convert for Neuron
                tensor = inp.contiguous() if not inp.is_contiguous() else inp
                converted.append(TypeConverter.convert_for_neuron(tensor))
            else:
                # Keep scalars as-is
                converted.append(inp)

        return tuple(converted)

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
    ) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
        """Prepare input dictionary for NEFF execution.

        Args:
            inputs: Preprocessed inputs
            device: Target device

        Returns:
            Tuple of (input_dict, list_arg_positions, list_arg_lengths)
            - input_dict: Dictionary mapping input names to tensors
            - list_arg_positions: Original positions of list/tuple arguments
            - list_arg_lengths: Number of elements in each list/tuple
        """

        def to_device(tensor: torch.Tensor) -> torch.Tensor:
            # Match device conversion semantics of CUDA, where input scalar (0-dimensional) CPU
            # tensors are automatically moved to the correct device
            if tensor.device != device and tensor.device.type == "cpu" and tensor.ndim == 0:
                tensor = tensor.to(device)
            return tensor

        input_dict = {}
        tensor_idx = 0
        list_arg_positions = []
        list_arg_lengths = []

        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                input_dict[f"input{tensor_idx}"] = TypeConverter.convert_for_neuron(to_device(inp))
                tensor_idx += 1
            elif inp is not None and i not in self.static_argnums:
                if isinstance(inp, list | tuple):
                    # Handle tuple of tensors and mixed structures for ops like index_put, index
                    list_arg_positions.append(i)
                    list_length = 0
                    for val in inp:
                        if isinstance(val, torch.Tensor):
                            input_dict[f"input{tensor_idx}"] = TypeConverter.convert_for_neuron(
                                to_device(val)
                            )
                            tensor_idx += 1
                            list_length += 1
                        elif isinstance(val, float | int):
                            scalar_tensor = torch.tensor(val, device=device)
                            scalar_tensor = TypeConverter.convert_for_neuron(scalar_tensor)
                            input_dict[f"input{tensor_idx}"] = scalar_tensor
                            tensor_idx += 1
                            list_length += 1
                    list_arg_lengths.append(list_length)
                else:
                    # Convert non-static scalars to tensors
                    scalar_tensor = torch.tensor(inp, device=device)
                    scalar_tensor = TypeConverter.convert_for_neuron(scalar_tensor)
                    input_dict[f"input{tensor_idx}"] = scalar_tensor
                    tensor_idx += 1
            # Static arguments are baked into compilation
        logger.debug(f"Prepared {len(input_dict)} inputs for execution")
        return input_dict, list_arg_positions, list_arg_lengths

    def extract_device(self, inputs: tuple) -> torch.device:
        """Extract device from input tensors.

        Args:
            inputs: Input tensors and scalars

        Returns:
            Device from first tensor, or "neuron"
        """
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

    def create_cpu_fallback_context(
        self,
        processed_inputs: tuple,
        list_arg_positions: list[int],
        list_arg_lengths: list[int],
    ) -> dict:
        """Create CPU fallback context dictionary from processed inputs.

        This method extracts static arguments and their positions based on
        static_argnums, and bundles them with list argument metadata into
        a dictionary suitable for passing to C++ bindings.

        Args:
            processed_inputs: Preprocessed input tensors and values
            list_arg_positions: Original positions of list/tuple arguments
            list_arg_lengths: Number of elements in each list/tuple

        Returns:
            Dictionary with keys:
            - static_args: List of static argument values
            - static_positions: List of int positions for static args
            - list_positions: List of int positions for list arguments
            - list_lengths: List of int lengths for each list argument
        """
        static_args = []
        static_positions = []

        # Extract static arguments based on static_argnums
        if self._static_argnums:
            for idx in self._static_argnums:
                if idx < len(processed_inputs):
                    static_args.append(processed_inputs[idx])
                    static_positions.append(idx)

        return {
            "static_args": static_args,
            "static_positions": static_positions,
            "list_positions": list_arg_positions,
            "list_lengths": list_arg_lengths,
        }
