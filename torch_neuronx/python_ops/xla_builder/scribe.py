"""
HLO Proto Builder - A Python DSL for constructing HLO (High Level Optimizer) protobuf modules.

This module provides a domain-specific language (DSL) for building HLO modules directly
from Python code. It allows developers to construct complex HLO computations using a
fluent, intuitive API that mirrors mathematical operations.

Key Components:
- HloScribe: Main builder class that converts Python functions to HLO modules
- HloShape: Represents tensor shapes and provides HLO operations (add, multiply, etc.)
- HloShapeType: Metaclass that auto-generates HLO operation methods
- ModuleResult: Container for the final HLO module protobuf

Example Usage:
    ```python
    def my_computation(scribe):
        # Create shape types
        f32 = scribe.f32  # 32-bit float type
        # Define parameters (inputs to the computation)
        x = f32[10, 20].Parameter(parameter_number=0)  # 10x20 float32 tensor
        y = f32[10, 20].Parameter(parameter_number=1)  # 10x20 float32 tensor

        # Perform operations
        sum_result = x.Add(y)           # Element-wise addition
        product = sum_result.Multiply(x) # Element-wise multiplication

        return product  # This becomes the root of the computation

    # Build the HLO module
    scribe = HloScribe()
    result = scribe(my_computation)

    # Access the protobuf module
    hlo_module = result.module_proto
    ```
"""

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ClassVar

from torch_neuronx.protos import hlo_pb2
from torch_neuronx.protos.xla import xla_data_pb2


def proto_set_attr(field, key: str | None, value: Any) -> None:
    """Set attributes in a protobuf field recursively.

    This utility function handles the complexity of setting various types of protobuf
    fields, including nested structures, repeated fields, and scalar values.

    Args:
        field: Protobuf field to modify (the parent object)
        key: Attribute key to set (None for direct field modification)
        value: Value to set, can be:
            - dict: Recursively sets nested attributes
            - list: Sets repeated field values
            - primitive types: Sets scalar values directly
            - protobuf objects: Uses CopyFrom for complex objects

    Examples:
        ```python
        # Setting a simple scalar field
        proto_set_attr(instruction, 'parameter_number', 0)

        # Setting nested dictionary attributes
        proto_set_attr(instruction, 'window', {
            'dimensions': [{'size': 3, 'stride': 1}]
        })

        # Setting repeated field
        proto_set_attr(instruction, 'operand_ids', [1, 2, 3])
        ```

    Notes:
        - Handles nested dictionaries recursively
        - Supports repeated fields (both composite and scalar)
        - Falls back to CopyFrom for unknown field types
        - Thread-safe for individual calls
    """
    attr = field if key is None else getattr(field, key)

    if isinstance(value, dict):
        # Recursively handle nested dictionary structures
        for sub_key, sub_value in value.items():
            proto_set_attr(attr, sub_key, sub_value)
        return

    if isinstance(attr, (int | float | str | bytes)):
        # Direct assignment for primitive types
        setattr(field, key, value)
    elif callable(getattr(attr, "extend", None)):  # Repeated field detected
        if callable(getattr(attr, "add", None)):
            # RepeatedComposite field - each element needs to be added and configured
            for item in value:
                proto_set_attr(attr.add(), None, item)
        else:
            # RepeatedScalar field - can assign entire list
            attr[:] = value
    else:
        # Complex protobuf object - use CopyFrom for deep copy
        # This handles cases where clients build proto objects directly
        attr.CopyFrom(value)


class HloScribe:
    """Main builder class for converting Python functions to HLO modules.

    HloScribe acts as the central coordinator for building HLO (High Level Optimizer)
    computations. It provides a context for creating HLO instructions and manages the
    overall module structure.

    The class follows a functional programming pattern where you define a computation
    as a Python function, and HloScribe converts it into an HLO module protobuf.

    Attributes:
        counter (int): Unique ID counter for instructions and computations
        module (HloModuleProto): The HLO module being constructed
        computation (HloComputation): The current computation being built
        context (ClassVar): Global context for nested operations

    Key Features:
        - Automatic type accessors (f32, s32, bf16, etc.)
        - Context management for nested computations
        - Unique ID generation for all HLO entities
        - Automatic program shape inference
        - Entry computation setup

    Example:
        ```python
        def matrix_multiply_add(scribe):
            f32 = scribe.f32

            # Define inputs
            a = f32[128, 256].Parameter(parameter_number=0)  # Matrix A
            b = f32[256, 512].Parameter(parameter_number=1)  # Matrix B
            bias = f32[128, 512].Parameter(parameter_number=2)  # Bias

            # define outputs
            matmul_out = f32[128, 512]


            # Perform computation
            matmul = matmul_out.Dot(a, b, dot_dimension_numbers={
                'lhs_contracting_dimensions': [1],
                'rhs_contracting_dimensions': [0]
            })
            result = matmul.Add(matmul, bias)

            return result

        scribe = HloScribe()
        module_result = scribe(matrix_multiply_add)
        ```
    """

    # Class-level context for managing nested operations
    context: ClassVar["HloScribe | None"] = None

    def __init__(self, start_counter: int = 0):
        """Initialize HloScribe with automatic type accessors.

        Args:
            start_counter: Starting value for unique ID generation (default: 0)

        Notes:
            - Creates type accessors for all XLA primitive types (f32, s32, bf16, etc.)
            - Initializes empty HLO module with one computation
            - Sets up counter for unique ID generation
        """
        # Create convenient type accessors (f32, s32, bf16, etc.)
        # This allows users to write: scribe.f32[10, 20] instead of more verbose syntax
        for name, value in xla_data_pb2.PrimitiveType.items():
            setattr(self, name.lower(), HloShape(self, value))

        self.counter = start_counter
        self._context = None

        # Initialize the HLO module structure
        self.module = hlo_pb2.HloModuleProto()
        self.computation = self.module.computations.add()

    @contextmanager
    def scribe_context(self):
        """Context manager for handling nested HLO scribe operations.

        This context manager ensures that nested computations (like those used in
        reduce operations or conditionals) can access the correct scribe instance.

        Usage:
            ```python
            with scribe.scribe_context():
                # HloShape operations will use this scribe
                result = some_hlo_function()
            ```

        Yields:
            None: Context is managed via class variable

        Notes:
            - Properly restores previous context on exit
            - Thread-safe for individual scribe instances
            - Required for operations with sub-computations
        """
        old_context = HloScribe.context
        HloScribe.context = self
        try:
            yield
        finally:
            HloScribe.context = old_context

    def __call__(self, func: Callable[["HloScribe"], "HloShape"]) -> "ModuleResult":
        """Convert a Python function to a complete HLO module.

        This is the main entry point for building HLO modules. The function should
        take an HloScribe instance and return an HloShape representing the final result.

        Args:
            func: Function that defines the HLO computation. Should accept HloScribe
                  as first argument and return HloShape as the computation result.

        Returns:
            ModuleResult: Wrapped HLO module protobuf ready for compilation/execution

        Process:
            1. Execute function within scribe context
            2. Extract parameters from generated instructions
            3. Set up program shape with parameter and result information
            4. Assign unique IDs and names to all components
            5. Configure entry computation settings

        Example:
            ```python
            def simple_add(scribe):
                f32 = scribe.f32
                x = f32[100].Parameter(parameter_number=0)
                y = f32[100].Parameter(parameter_number=1)
                out = f32[100]
                return out.Add(x, y)

            scribe = HloScribe()
            result = scribe(simple_add)

            # result.module_proto contains the complete HLO module
            print(f"Module name: {result.module_proto.name}")
            print(f"Instructions: {len(result.module_proto.computations[0].instructions)}")
            ```

        Notes:
            - Parameters are automatically sorted by parameter_number
            - Program shape is inferred from parameters and result
            - Entry computation is properly configured
            - All IDs and names are automatically assigned
        """
        # Execute the user-defined function within our context
        with self.scribe_context():
            result = func(self)

        func_name = func.__name__

        # Extract and sort parameters by their parameter numbers
        # Parameters are the inputs to the HLO computation
        parameters = [inst for inst in self.computation.instructions if inst.opcode == "parameter"]
        parameters.sort(key=lambda x: x.parameter_number)

        # Set up the program shape - this describes the "signature" of the computation
        program_shape = self.computation.program_shape

        # Configure parameter information
        for i, param in enumerate(parameters):
            param.name = f"p{i}.{param.id}"  # Standard parameter naming
            # Add parameter shape to program signature
            program_shape.parameters.add().CopyFrom(param.shape)
            program_shape.parameter_names.append(f"p{i}")

        # Set the result shape and root instruction
        program_shape.result.CopyFrom(result.shape_proto)
        self.computation.root_id = result.instruction.id

        # Assign unique IDs and names for the computation and module
        comp_id = self.get_program_counter()
        self.computation.id = comp_id
        self.computation.name = f"{func_name}.{comp_id}"

        # Configure module-level settings
        self.module.id = self.module.entry_computation_id = comp_id
        self.module.name = self.module.entry_computation_name = self.computation.name
        self.module.host_program_shape.CopyFrom(program_shape)

        return ModuleResult(self.module)

    def get_program_counter(self) -> int:
        """Generate the next unique ID for HLO entities.

        Returns:
            int: Next unique identifier

        Notes:
            - Thread-safe for individual scribe instances
            - Used for instructions, computations, and modules
            - Ensures all HLO entities have unique identifiers
        """
        self.counter += 1
        return self.counter


class HloShapeType(type):
    """Metaclass for HloShape that auto-generates HLO operation methods.

    This metaclass is responsible for creating all the HLO operation methods on the
    HloShape class. Instead of manually defining hundreds of methods, it programmatically
    generates them based on the operation definitions in op_code_map.

    The metaclass handles:
    - Method generation for all HLO operations (Add, Multiply, Dot, etc.)
    - Operand count validation
    - Sub-computation management for complex operations
    - Replica group handling for distributed operations
    - Parameter validation and processing

    Class Variables:
        is_hlo_opcode_variadic (int): Marker for operations with variable operand count
        constant_value (str): Key for constant value operations
        op_code_map (dict): Complete mapping of HLO operations to their properties
        opcode_with_sub_computation (set): Operations requiring sub-computations
        opcode_with_replica_groups (set): Distributed operations using replica groups

    Generated Methods:
        Each entry in op_code_map becomes a method on HloShape:
        - Add(other) -> element-wise addition
        - Multiply(a, b) -> element-wise multiplication
        - Dot(a, b, **config) -> matrix multiplication
        - Reduce(operands, init_value, to_apply=func) -> reduction operation
        - etc.

    Example of Generated Usage:
        ```python
        f32 = scribe.f32
        a = f32[10, 20].Parameter(parameter_number=0)
        b = f32[10, 20].Parameter(parameter_number=1)
        out = f32[10, 20]

        # These methods are auto-generated by this metaclass:
        sum_result = out.Add(a, b)                    # Binary operation
        abs_result = out.Abs(a)                     # Unary operation
        ```
    """

    # Special marker for operations that accept variable number of operands
    is_hlo_opcode_variadic: int = -1
    constant_value: str = "constant_value"

    # Complete mapping of HLO operations
    # Format: "MethodName": ("hlo-opcode", operand_count)
    # operand_count = -1 means variable number of operands
    op_code_map: ClassVar[dict[str, tuple[str, int]]] = {
        # Unary operations (1 operand)
        "Abs": ("abs", 1),
        "Ceil": ("ceil", 1),
        "Floor": ("floor", 1),
        "Sqrt": ("sqrt", 1),
        "Cbrt": ("cbrt", 1),
        "Exp": ("exponential", 1),
        "Expm1": ("exponential-minus-one", 1),
        "Log": ("log", 1),
        "Log1p": ("log-plus-one", 1),
        "Cos": ("cosine", 1),
        "Sin": ("sine", 1),
        "Tanh": ("tanh", 1),
        "Logistic": ("logistic", 1),
        "Sign": ("sign", 1),
        "Negate": ("negate", 1),
        "Not": ("not", 1),
        "Rsqrt": ("rsqrt", 1),
        "IsFinite": ("is-finite", 1),
        "Imag": ("imag", 1),
        "Real": ("real", 1),
        "PopulationCount": ("popcnt", 1),
        "Clz": ("count-leading-zeros", 1),
        "RoundNearestAfz": ("round-nearest-afz", 1),
        "RoundNearestEven": ("round-nearest-even", 1),
        # Binary operations (2 operands)
        "Add": ("add", 2),
        "Subtract": ("subtract", 2),
        "Multiply": ("multiply", 2),
        "Divide": ("divide", 2),
        "Remainder": ("remainder", 2),
        "Maximum": ("maximum", 2),
        "Minimum": ("minimum", 2),
        "Power": ("power", 2),
        "Atan2": ("atan2", 2),
        "Complex": ("complex", 2),
        "And": ("and", 2),
        "Or": ("or", 2),
        "Xor": ("xor", 2),
        "ShiftLeft": ("shift-left", 2),
        "ShiftRightArithmetic": ("shift-right-arithmetic", 2),
        "ShiftRightLogical": ("shift-right-logical", 2),
        "Compare": ("compare", 2),
        "Dot": ("dot", 2),
        "Convolution": ("convolution", 2),
        # Ternary operations (3 operands)
        "Select": ("select", 3),  # condition ? true_value : false_value
        "Clamp": ("clamp", 3),  # clamp(min, value, max)
        "TupleSelect": ("tuple-select", 3),
        # Shape manipulation (1 operand)
        "Bitcast": ("bitcast", 1),
        "BitcastConvert": ("bitcast-convert", 1),
        "Broadcast": ("broadcast", 1),
        "Convert": ("convert", 1),
        "Copy": ("copy", 1),
        "Reshape": ("reshape", 1),
        "Reverse": ("reverse", 1),
        "Slice": ("slice", 1),
        "Transpose": ("transpose", 1),
        "GetTupleElement": ("get-tuple-element", 1),
        # Special operations (0 operands)
        "Constant": ("constant", 0),
        "Parameter": ("parameter", 0),
        "Iota": ("iota", 0),
        "ReplicaId": ("replica-id", 0),
        "PartitionId": ("partition-id", 0),
        "RngGetAndUpdateState": ("rng-get-and-update-state", 0),
        # Variable operand operations
        "Tuple": ("tuple", is_hlo_opcode_variadic),
        "Concatenate": ("concatenate", is_hlo_opcode_variadic),
        "DynamicSlice": ("dynamic-slice", is_hlo_opcode_variadic),
        "DynamicUpdateSlice": ("dynamic-update-slice", is_hlo_opcode_variadic),
        "DynamicReshape": ("dynamic-reshape", is_hlo_opcode_variadic),
        "Rng": ("rng", is_hlo_opcode_variadic),
        # Distributed/collective operations
        "AllGather": ("all-gather", is_hlo_opcode_variadic),
        "AllReduce": ("all-reduce", is_hlo_opcode_variadic),
        "AllReduceScatter": ("all-reduce-scatter", is_hlo_opcode_variadic),
        "AllToAll": ("all-to-all", is_hlo_opcode_variadic),
        "CollectivePermute": ("collective-permute", is_hlo_opcode_variadic),
        # Operations with sub-computations
        "Call": ("call", is_hlo_opcode_variadic),
        "Map": ("map", is_hlo_opcode_variadic),
        "Reduce": ("reduce", is_hlo_opcode_variadic),
        "ReduceWindow": ("reduce-window", is_hlo_opcode_variadic),
        "Sort": ("sort", is_hlo_opcode_variadic),
        "Scatter": ("scatter", 3),
        "SelectAndScatter": ("select-and-scatter", 3),
        # Advanced operations
        "BatchNormTraining": ("batch-norm-training", 3),
        "BatchNormInference": ("batch-norm-inference", 5),
        "BatchNormGrad": ("batch-norm-grad", 5),
        "Cholesky": ("cholesky", 1),
        "TriangularSolve": ("triangular-solve", 2),
        "Fft": ("fft", 1),
        "Gather": ("gather", 2),
        "Pad": ("pad", 2),
        "While": ("while", 1),
        "Conditional": ("conditional", is_hlo_opcode_variadic),
        # Communication operations
        "Send": ("send", 2),
        "Recv": ("recv", 1),
        "Infeed": ("infeed", 1),
        "Outfeed": ("outfeed", 2),
        # Async operations
        "AllReduceStart": ("all-reduce-start", is_hlo_opcode_variadic),
        "AllReduceDone": ("all-reduce-done", 1),
        "CollectivePermuteStart": ("collective-permute-start", is_hlo_opcode_variadic),
        "CollectivePermuteDone": ("collective-permute-done", 1),
        "SendDone": ("send-done", 1),
        "RecvDone": ("recv-done", 1),
        "CopyStart": ("copy-start", 1),
        "CopyDone": ("copy-done", 1),
        # Other operations
        "AddDependency": ("add-dependency", 2),
        "AfterAll": ("after-all", is_hlo_opcode_variadic),
        "Domain": ("domain", 1),
        "Fusion": ("fusion", is_hlo_opcode_variadic),
        "GetDimensionSize": ("get-dimension-size", 1),
        "SetDimensionSize": ("set-dimension-size", 2),
        "ReducePrecision": ("reduce-precision", 1),
        "ReduceScatter": ("reduce-scatter", 1),
        "Trace": ("trace", 1),
        "CustomCall": ("custom-call", is_hlo_opcode_variadic),
        "RngBitGenerator": ("rng-bit-generator", 1),
    }

    # Operations that require sub-computations (lambda functions)
    # Example: Reduce(data, init_value, to_apply=lambda x, y: x.Add(y))
    opcode_with_sub_computation: ClassVar[set] = {
        "all-reduce",  # Custom reduction function
        "call",  # Function to call
        "map",  # Function to map over elements
        "reduce",  # Reduction function
        "reduce-scatter",  # Distributed reduction function
        "reduce-window",  # Window reduction function
        "scatter",  # Update function for scatter
        "select-and-scatter",  # Selection and update functions
        "sort",  # Comparison function
    }

    # Operations that work with replica groups for distributed execution
    opcode_with_replica_groups: ClassVar[set] = {
        "all-gather",
        "all-reduce",
        "all-to-all",
        "reduce-scatter",
    }

    def __new__(cls, name: str, bases: tuple, dct: dict) -> type:
        """Create new HloShape class with auto-generated operation methods.

        This method runs when the HloShape class is created and automatically
        generates methods for all operations defined in op_code_map.

        Args:
            name: Class name being created
            bases: Base classes
            dct: Class dictionary to populate with methods

        Returns:
            type: New class with all generated operation methods

        Notes:
            - Creates one method per entry in op_code_map
            - Each generated method handles operand validation and instruction creation
            - Methods are added to the class dictionary before class creation
        """
        # Generate a method for each operation in the map
        for fname, opdef in cls.op_code_map.items():
            dct[fname] = cls.gen_api(fname, opdef)
        return super().__new__(cls, name, bases, dct)

    @classmethod
    def gen_api(cls, fname: str, opdef: tuple[str, int]) -> Callable:
        """Generate an API method for a specific HLO operation.

        This method creates the actual implementation for each HLO operation method.
        The generated method handles operand validation, instruction creation, and
        special cases like sub-computations and replica groups.

        Args:
            fname: Function name to generate (e.g., "Add", "Multiply")
            opdef: Tuple of (opcode, arity) defining the operation

        Returns:
            Callable: Generated method that creates HLO instructions

        Generated Method Signature:
            ```python
            def operation_name(self, *operands, **kwargs) -> HloShape:
                # Validates operands, creates instruction, handles special cases
                return new_hlo_shape_with_instruction
            ```

        Special Handling:
            - Sub-computations: Operations like Reduce require lambda functions
            - Replica groups: Distributed operations need process group configuration
            - Variadic operations: Some operations accept variable operand counts
            - Constant values: Special handling for constant operations

        """
        opcode, arity = opdef

        def api(self, *operands, **kwargs) -> "HloShape":
            """Generated API method for HLO operation.

            Args:
                *operands: Input operands for the operation
                **kwargs: Additional configuration parameters

            Returns:
                HloShape: New shape with the operation instruction

            Raises:
                ValueError: If operand count doesn't match expected arity
                NotImplementedError: For unsupported constant operations
            """
            # Create a copy of the current shape to avoid mutation
            self = self.clone()

            # Validate operand count for fixed-arity operations
            if arity != cls.is_hlo_opcode_variadic and len(operands) != arity:
                raise ValueError(
                    f"{fname} expects {arity} operands, but received {len(operands)}. "
                    f"Operands provided: {[type(op).__name__ for op in operands]}"
                )

            # Create new HLO instruction
            inst = self.scribe.computation.instructions.add()
            inst.opcode = opcode
            inst.id = self.scribe.get_program_counter()
            inst.name = f"{opcode}.{inst.id}"

            # Set operand references (instructions that provide inputs)
            inst.operand_ids[:] = [op.instruction.id for op in operands]

            # Copy the shape (output type/dimensions)
            inst.shape.CopyFrom(self.shape_proto)

            # Handle operations that require sub-computations
            if opcode in cls.opcode_with_sub_computation:
                if "to_apply" not in kwargs:
                    raise ValueError(
                        f"{fname} requires 'to_apply' parameter with a function. "
                        f"Example: {fname}(..., to_apply=lambda x, y: x.Add(y))"
                    )

                # Create sub-computation for the lambda function
                to_apply = kwargs.pop("to_apply")
                sub_scribe = HloScribe(self.scribe.get_program_counter())
                sub_scribe(to_apply)
                sub_computation = sub_scribe.computation
                sub_computation.name = f"{inst.name}.{sub_computation.name}"

                # Add sub-computation to module (at beginning for topological order)
                self.scribe.module.computations.insert(0, sub_computation)
                inst.called_computation_ids.append(sub_computation.id)

                # Update counter to maintain uniqueness
                self.scribe.counter = sub_computation.id

            # Handle distributed operations with replica groups
            if opcode in cls.opcode_with_replica_groups:
                if "replica_groups" not in kwargs:
                    raise ValueError(
                        f"{fname} requires 'replica_groups' parameter. "
                        f"Example: {fname}(..., replica_groups=[[0, 1], [2, 3]])"
                    )

                # Convert replica groups to proper protobuf format
                replica_groups = []
                for group in kwargs["replica_groups"]:
                    if not isinstance(group, xla_data_pb2.ReplicaGroup):
                        # Convert list to ReplicaGroup protobuf
                        group = xla_data_pb2.ReplicaGroup(replica_ids=group)
                    replica_groups.append(group)
                kwargs["replica_groups"] = replica_groups

            # Set additional attributes from kwargs
            for key, value in kwargs.items():
                if key == cls.constant_value:
                    raise NotImplementedError(
                        "Constant value operations are not yet supported. "
                        "Use Parameter operations for inputs instead."
                    )
                # Use utility function to handle complex protobuf attribute setting
                proto_set_attr(inst, key, value)

            # Update this shape to reference the new instruction
            self.instruction = inst
            return self

        # Set method name and documentation for better debugging
        api.__name__ = fname
        api.__doc__ = f"Generate HLO {opcode} instruction with {arity} operands."

        return api


class HloShape(metaclass=HloShapeType):
    """Represents an HLO tensor shape and provides fluent operation API.

    HloShape is the core abstraction for building HLO computations. It represents
    both the shape/type information of tensors and provides methods for all HLO
    operations. The class uses a fluent API where operations return new HloShape
    instances, allowing for chained operations.

    Key Features:
        - Fluent API: operations can be chained (a.Add(b).Multiply(c))
        - Shape inference: automatically tracks tensor shapes through operations
        - Type safety: maintains element type information (f32, s32, etc.)
        - Dimensional analysis: tracks tensor dimensions and layouts
        - Operation generation: all HLO operations available as methods

    Attributes:
        scribe (HloScribe): Reference to the HLO builder context
        shape_proto (ShapeProto): Protobuf describing tensor shape and type
        instruction (HloInstruction): The instruction that produces this shape

    Usage Patterns:
        ```python
        # Creating typed shapes
        f32 = scribe.f32           # Get float32 type
        shape = f32[100, 200]        # Create 100x200 float32 tensor shape

        # Creating parameters (inputs)
        x = f32[100, 200].Parameter(parameter_number=0)
        y = f32[100, 200].Parameter(parameter_number=1)

        # Chaining operations
        result = shape.Sqrt(shape.Multiply(shape.Add(x, y), x))

        ```

    Shape Creation:
        ```python
        # Scalar shapes
        scalar = f32[]  # Scalar float32

        # Vector shapes
        vector = f32[1000]  # 1D vector of 1000 elements

        # Matrix shapes
        matrix = f32[128, 256]  # 2D matrix 128x256

        # Higher dimensional tensors
        tensor_4d = f32[8, 3, 224, 224]  # Batch of RGB images

        # Tuple shapes (multiple outputs)
        tuple_shape = scribe.f32(  # Creates tuple shape
            f32[10, 10],           # First element
            s32[10]                # Second element
        )
        ```
    """

    def __init__(self, scribe: HloScribe, element_type: xla_data_pb2.PrimitiveType):
        """Initialize an HLO shape with specified element type.

        Args:
            scribe: The HLO scribe instance managing this computation
            element_type: XLA primitive type (F32, S32, BF16, etc.)

        Notes:
            - Creates empty shape prototype with specified element type
            - No dimensions initially (use __getitem__ to add dimensions)
            - No associated instruction initially (created by operations)
        """
        self.scribe = scribe
        self.shape_proto = xla_data_pb2.ShapeProto()
        self.shape_proto.element_type = element_type
        self.instruction = None

    def __getitem__(self, dimensions: int | list[int] | tuple[int, ...]) -> "HloShape":
        """Define tensor dimensions for the shape.

        This method is called when using bracket notation to specify tensor dimensions.
        It creates a new HloShape with the specified dimensions and proper layout.

        Args:
            dimensions: Tensor dimensions. Can be:
                - int: Creates 1D tensor [dimensions]
                - list/tuple: Creates multi-dimensional tensor

        Returns:
            HloShape: New shape with specified dimensions and row-major layout

        Examples:
            ```python
            # Scalar (no dimensions)
            scalar = f32[]

            # 1D vector
            vector = f32[1000]

            # 2D matrix
            matrix = f32[128, 256]

            # 3D tensor
            tensor_3d = f32[8, 128, 256]

            # 4D tensor (common for images: batch, channels, height, width)
            images = f32[32, 3, 224, 224]
            ```

        Notes:
            - Automatically sets row-major layout (C-style memory order)
            - All dimensions are initially non-dynamic
            - Creates independent copy of the shape
        """
        new_shape = HloShape(self.scribe, self.shape_proto.element_type)

        # Normalize dimensions to list
        if not isinstance(dimensions, (list | tuple)):
            dimensions = [dimensions]

        # Set dimensions and dynamic flags
        new_shape.shape_proto.dimensions[:] = dimensions
        new_shape.shape_proto.is_dynamic_dimension[:] = [False] * len(dimensions)

        # Set default row-major layout (last dimension varies fastest)
        # For a 3D tensor [8, 128, 256], layout [2, 1, 0] means:
        # - dimension 2 (256) varies fastest in memory
        # - dimension 1 (128) varies next
        # - dimension 0 (8) varies slowest
        if dimensions:  # Only set layout for non-scalar tensors
            new_shape.shape_proto.layout.minor_to_major[:] = list(reversed(range(len(dimensions))))

        return new_shape

    def __call__(self, *tensors: list["HloShape"]) -> "HloShape":
        """Create a tuple shape from multiple tensor shapes.

        This method allows creating tuple types that can hold multiple values
        of potentially different types and shapes. Useful for operations that
        return multiple outputs.

        Args:
            *tensors: Variable number of HloShape instances to combine

        Returns:
            HloShape: New tuple shape containing all input shapes


        Notes:
            - Preserves the element type of the original shape (not typically used)
            - Each tensor in the tuple maintains its own shape and type
            - Tuple shapes can be nested for complex data structures
        """
        shape = HloShape(self.scribe, self.shape_proto.element_type)

        # Add each input tensor as a tuple element
        for tensor in tensors:
            shape.shape_proto.tuple_shapes.add().CopyFrom(tensor.shape_proto)

        return shape

    def clone(self) -> "HloShape":
        """Create a deep copy of the current shape.

        This method is essential for the immutable operation semantics of HloShape.
        Each operation creates a new shape rather than modifying existing ones.

        Returns:
            HloShape: New shape with identical properties but independent state

        Notes:
            - Copies shape prototype (dimensions, type, layout)
            - Preserves scribe reference for continued operation
            - Does not copy instruction reference (set by operations)
            - Thread-safe for individual shape instances

        Usage:
            ```python
            # Typically used internally by operation methods
            original = f32[10, 20]
            copy = original.clone()  # Independent copy

            # Operations use clone to avoid mutation
            result = original.Add(other)  # original unchanged
            ```
        """
        shape = HloShape(self.scribe, self.shape_proto.element_type)
        shape.shape_proto.CopyFrom(self.shape_proto)
        return shape


class ModuleResult:
    """Container for completed HLO module protobuf.

    This class wraps the final HLO module protobuf that results from building
    a computation with HloScribe. It provides access to the complete module
    structure including computations, instructions, and metadata.

    Attributes:
        module_proto (HloModuleProto): The complete HLO module protobuf

    The module contains:
        - Entry computation: Main computation to execute
        - Sub-computations: Any nested computations (for reduce, map, etc.)
        - Program shape: Input and output signatures
        - Module metadata: Names, IDs, and configuration

    Usage:
        ```python
        def my_computation(scribe):
            f32 = scribe.f32
            x = f32[100].Parameter(parameter_number=0)
            return f32[100].Abs(x)

        scribe = HloScribe()
        result = scribe(my_computation)

        # Access the protobuf module
        hlo_module = result.module_proto

        # Inspect module properties
        print(f"Module name: {hlo_module.name}")
        print(f"Entry computation: {hlo_module.entry_computation_name}")
        print(f"Parameter count: {len(hlo_module.host_program_shape.parameters)}")

        # Access computations and instructions
        entry_comp = hlo_module.computations[0]
        print(f"Instruction count: {len(entry_comp.instructions)}")

        # Serialize for compilation or storage
        serialized_bytes = hlo_module.SerializeToString()
        ```

    Integration:
        The module_proto can be:
        - Passed to XLA compiler for optimization and execution
        - Serialized for storage or network transmission
        - Analyzed for debugging or visualization
        - Modified for advanced use cases
    """

    def __init__(self, module_proto: hlo_pb2.HloModuleProto):
        """Initialize with completed HLO module protobuf.

        Args:
            module_proto: Complete HLO module protobuf from HloScribe
        """
        self.module_proto = module_proto

    def __str__(self) -> str:
        """String representation showing module summary."""
        comp_count = len(self.module_proto.computations)
        if comp_count > 0:
            inst_count = len(self.module_proto.computations[0].instructions)
            return (
                f"HloModule({self.module_proto.name}, {comp_count} computations, "
                f"{inst_count} instructions)"
            )
        return f"HloModule({self.module_proto.name}, empty)"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"ModuleResult(name='{self.module_proto.name}', id={self.module_proto.id})"
