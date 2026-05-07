"""
Implementation of custom MLIR passes for Neuron device.
"""

import re

import numpy as np
from torch_mlir.dialects.stablehlo._stablehlo_ops_gen import ConstantOp
from torch_mlir.ir import (
    F32Type,
    InsertionPoint,
    RankedTensorType,
    Type,
    TypeAttr,
)


class DTypeConstantConverter:
    """
    Converts f64 constants to f32 in MLIR module
    """

    def _need_op_conversion(self, op):
        """Check if an operation needs f64 to f32 conversion.

        Examines the operation's result types to determine if any
        contain f64 element types that need conversion.

        Args:
            op: MLIR operation to check.

        Returns:
            bool: True if any result has f64 element type.
        """
        # Check if any result needs conversion
        for result in op.results:
            result_type = result.type
            if hasattr(result_type, "element_type") and str(result_type.element_type) in [
                "f64",
            ]:
                return True
        return False

    def _get_new_type(self, op, result_type):
        """Get the converted type for f64 to f32 conversion.

        Args:
            op: MLIR operation (used for location).
            result_type: Original MLIR type with f64 element type.

        Returns:
            Type: New MLIR type with f32 element type.

        Raises:
            TypeError: If element type is not f64.
        """
        elem_type = str(result_type.element_type)
        if elem_type == "f64":
            new_elem_type = F32Type.get()
        else:
            raise TypeError(f"Invalid dtype ({elem_type}) for conversion")

        if hasattr(result_type, "shape"):
            new_type = RankedTensorType.get(result_type.shape, new_elem_type, loc=op.location)
        else:
            new_type = new_elem_type
        return new_type

    def _convert_constant_op(self, op):
        """Convert a constant operation from f64 to f32.

        Creates a new DenseElementsAttr with f32 dtype from the original
        f64 constant value.

        Args:
            op: StableHLO constant operation with f64 type.

        Returns:
            DenseElementsAttr: New attribute with f32 values.

        Raises:
            TypeError: If the constant is not f64 type.
        """
        # Get the value attribute
        value_attr = op.attributes["value"]
        result_type = op.results[0].type
        new_type = self._get_new_type(op, result_type)

        # Determine np data type types
        elem_type = str(result_type.element_type)
        if elem_type == "f64":
            target_dtype = np.float32
        else:
            raise TypeError(f"Invalid dtype ({elem_type}) for conversion")

        # Convert the DenseElementsAttr
        new_value_attr = self.convert_dense_attr(value_attr, new_type, target_dtype)

        return new_value_attr

    def convert_dense_attr(self, old_attr, new_type, target_dtype):
        """Convert a DenseElementsAttr to a target numpy dtype.

        Extracts values from the attribute, converts to target dtype,
        and creates a new attribute with the converted values.

        Args:
            old_attr: Original DenseElementsAttr.
            new_type: Target MLIR type for the new attribute.
            target_dtype: Target numpy dtype (e.g., np.float32).

        Returns:
            DenseElementsAttr: New attribute with converted values.
        """
        # Extract numpy array from DenseElementsAttr
        np_array = np.array(old_attr)

        # Convert to target dtype
        np_array_converted = np_array.astype(target_dtype)

        old_attr_type = type(old_attr)

        # Create new Attribute
        new_attr = old_attr_type.get(np_array_converted, type=new_type)
        return new_attr

    def _convert_op_types(self, op):
        """Convert operation result types from f64 to f32 in place.

        Modifies the operation's result types directly without
        creating new operations.

        Args:
            op: MLIR operation to modify.
        """
        for result in op.results:
            new_type = self._get_new_type(op, result.type)
            result.set_type(new_type)

    def _replace_constant(self, old_op, new_value):
        """Replace an f64 constant operation with an f32 constant.

        Creates a new constant operation with the converted value,
        replaces all uses of the old operation, and erases the old operation.

        Args:
            old_op: Original constant operation to replace.
            new_value: New DenseElementsAttr with f32 values.
        """
        # Create new constant operation before the old one
        with InsertionPoint(old_op):
            # Use the old operation's location
            loc = old_op.location
            new_op = ConstantOp(new_value, loc=loc)

            # Replace all uses
            old_op.results[0].replace_all_uses_with(new_op.result)

        # Erase old operation
        old_op.operation.erase()

    def run(self, module):
        """
        Run the conversion pass on the entire module
        """
        with module.context:
            ops_to_replace = []
            funcs_to_update = []

            # Walk through all operations
            def walk_op(op):
                # Convert function signatures
                if op.operation.name == "func.func" and self._needs_func_conversion(op):
                    funcs_to_update.append(op)

                elif self._need_op_conversion(op):
                    if op.operation.name == "stablehlo.constant":
                        # Convert f64 ConstantOp by replacing the op
                        conversion_result = self._convert_constant_op(op)
                        ops_to_replace.append((op, conversion_result))
                    else:
                        # Convert other ops inplace
                        self._convert_op_types(op)

                # TODO: Address op.attributes

                # Recurse into regions
                for region in op.regions:
                    for block in region:
                        # convert reduction regions
                        for arg in block.arguments:
                            arg_type_str = str(arg.type)
                            if "f64" in arg_type_str:
                                new_type_str = arg_type_str.replace("f64", "f32")

                                new_type = Type.parse(new_type_str, context=op.context)
                                arg.set_type(new_type)

                        for inner_op in block.operations:
                            walk_op(inner_op)

            # Walk the module
            for op in module.body.operations:
                walk_op(op)

            # Replace operations in reverse order to avoid invalidation
            for old_op, new_value in reversed(ops_to_replace):
                self._replace_constant(old_op, new_value)

            # Update function signatures after all operations are converted
            for func_op in funcs_to_update:
                self._convert_func_signature(func_op)

        return module

    def _needs_func_conversion(self, func_op):
        """Check if a function signature contains f64 types.

        Args:
            func_op: MLIR function operation.

        Returns:
            bool: True if function type string contains 'f64'.
        """
        func_type = func_op.attributes["function_type"]
        type_str = str(func_type)

        # Convert both standalone f64 and tensor<...f64>
        return bool(re.search(r"(f64)", type_str))

    def _convert_func_signature(self, func_op):
        """Convert f64 to f32 in a function's signature.

        Updates the function_type attribute and block argument types
        to use f32 instead of f64.

        Args:
            func_op: MLIR function operation to modify.
        """
        old_func_type = func_op.attributes["function_type"]
        type_str = str(old_func_type)

        # Replace all f64 occurrences, including inside tensor types
        new_type_str = re.sub(r"f64", "f32", type_str)
        new_func_type = TypeAttr.parse(new_type_str, context=func_op.context)
        func_op.attributes["function_type"] = new_func_type

        # Also update block argument types
        self._convert_block_args(func_op)

    def _convert_block_args(self, func_op):
        """Convert f64 block arguments to f32.

        Iterates through all blocks in the function and converts
        any f64-typed arguments to f32.

        Args:
            func_op: MLIR function operation containing blocks.
        """
        for region in func_op.regions:
            for block in region:
                for i, arg in enumerate(block.arguments):
                    arg_type = str(arg.type)
                    if arg_type == "f64":
                        new_type = F32Type.get()
                        block.arguments[i].set_type(new_type)


dtype_converter = DTypeConstantConverter()


def dtype_conversion_pass(module):
    """
    Main entry point: convert all f64 constants to f32
    """
    return dtype_converter.run(module)
