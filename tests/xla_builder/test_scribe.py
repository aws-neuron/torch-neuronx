import os
import tempfile

import pytest

from torch_neuronx.protos import hlo_pb2
from torch_neuronx.protos.xla import xla_data_pb2
from torch_neuronx.python_ops.xla_builder.scribe import HloScribe, HloShape, proto_set_attr


class TestHloScribe:
    """Test cases for HloScribe functionality."""

    def test_simple_parameter_creation(self):
        """Test creating simple parameters."""

        def func(scribe):
            f32 = scribe.f32
            return f32[10, 20].Parameter(parameter_number=0)

        scribe = HloScribe()
        result = scribe(func)

        assert result.module_proto.name == "func.2"
        assert len(result.module_proto.computations) == 1

        comp = result.module_proto.computations[0]
        assert len(comp.instructions) == 1

        inst = comp.instructions[0]
        assert inst.opcode == "parameter"
        assert inst.parameter_number == 0
        assert inst.shape.element_type == xla_data_pb2.PrimitiveType.F32
        assert list(inst.shape.dimensions) == [10, 20]

    def test_simple_add_operation(self):
        """Test simple addition of two parameters."""

        def simple_add(scribe):
            f32 = scribe.f32
            lhs = f32[16, 6].Parameter(parameter_number=0)
            rhs = f32[16, 6].Parameter(parameter_number=1)
            return f32[16, 6].Add(lhs, rhs)

        scribe = HloScribe()
        result = scribe(simple_add)

        comp = result.module_proto.computations[0]
        assert len(comp.instructions) == 3  # 2 parameters + 1 add

        # Check parameters
        params = [inst for inst in comp.instructions if inst.opcode == "parameter"]
        assert len(params) == 2
        assert params[0].parameter_number == 0
        assert params[1].parameter_number == 1

        # Check add operation
        add_ops = [inst for inst in comp.instructions if inst.opcode == "add"]
        assert len(add_ops) == 1
        add_op = add_ops[0]
        assert len(add_op.operand_ids) == 2
        assert add_op.shape.element_type == xla_data_pb2.PrimitiveType.F32
        assert list(add_op.shape.dimensions) == [16, 6]

    def test_multiple_operations(self):
        """Test chaining multiple operations."""

        def multi_ops(scribe):
            f32 = scribe.f32
            x = f32[10].Parameter(parameter_number=0)
            y = f32[10].Parameter(parameter_number=1)
            z = f32[10].Parameter(parameter_number=2)

            sum_xy = f32[10].Add(x, y)
            prod = f32[10].Multiply(sum_xy, z)
            return f32[10].Abs(prod)

        scribe = HloScribe()
        result = scribe(multi_ops)

        comp = result.module_proto.computations[0]

        # Should have 3 parameters + 3 operations
        assert len(comp.instructions) == 6

        # Check operation types
        opcodes = [inst.opcode for inst in comp.instructions]
        assert opcodes.count("parameter") == 3
        assert opcodes.count("add") == 1
        assert opcodes.count("multiply") == 1
        assert opcodes.count("abs") == 1

    def test_dot_operation_with_attributes(self):
        """Test dot operation with dimension numbers."""

        def dot_func(scribe):
            f32 = scribe.f32
            lhs = f32[16, 6].Parameter(parameter_number=0)
            rhs = f32[16, 8].Parameter(parameter_number=1)

            dot_dims = {"lhs_contracting_dimensions": [0], "rhs_contracting_dimensions": [0]}
            return f32[6, 8].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)

        scribe = HloScribe()
        result = scribe(dot_func)

        comp = result.module_proto.computations[0]
        dot_ops = [inst for inst in comp.instructions if inst.opcode == "dot"]
        assert len(dot_ops) == 1

        dot_op = dot_ops[0]
        assert len(dot_op.dot_dimension_numbers.lhs_contracting_dimensions) == 1
        assert dot_op.dot_dimension_numbers.lhs_contracting_dimensions[0] == 0
        assert len(dot_op.dot_dimension_numbers.rhs_contracting_dimensions) == 1
        assert dot_op.dot_dimension_numbers.rhs_contracting_dimensions[0] == 0

    def test_broadcast_operation(self):
        """Test broadcast operation with dimensions."""

        def broadcast_func(scribe):
            f32 = scribe.f32
            bias = f32[8].Parameter(parameter_number=0)
            return f32[6, 8].Broadcast(bias, dimensions=[1])

        scribe = HloScribe()
        result = scribe(broadcast_func)

        comp = result.module_proto.computations[0]
        broadcast_ops = [inst for inst in comp.instructions if inst.opcode == "broadcast"]
        assert len(broadcast_ops) == 1

        broadcast_op = broadcast_ops[0]
        assert list(broadcast_op.dimensions) == [1]
        assert list(broadcast_op.shape.dimensions) == [6, 8]

    def test_all_reduce_with_subcomputation(self):
        """Test AllReduce operation with sub-computation."""

        def reducer(scribe):
            f32 = scribe.f32
            p0 = f32.Parameter(parameter_number=0)
            p1 = f32.Parameter(parameter_number=1)
            return f32.Add(p0, p1)

        def func_all_reduce(scribe):
            f32 = scribe.f32
            p0 = f32[16, 6].Parameter(parameter_number=0)
            return f32[16, 6].AllReduce(p0, replica_groups=[[0, 1, 2, 3]], to_apply=reducer)

        scribe = HloScribe()
        result = scribe(func_all_reduce)

        # Should have 2 computations: main + reducer
        assert len(result.module_proto.computations) == 2

        # Check main computation
        main_comp = result.module_proto.computations[1]
        all_reduce_ops = [inst for inst in main_comp.instructions if inst.opcode == "all-reduce"]
        assert len(all_reduce_ops) == 1

        all_reduce_op = all_reduce_ops[0]
        assert len(all_reduce_op.called_computation_ids) == 1

        # Check sub-computation exists
        sub_comp_id = all_reduce_op.called_computation_ids[0]
        sub_comp = next(comp for comp in result.module_proto.computations if comp.id == sub_comp_id)
        assert sub_comp is not None

        # Check sub-computation has add operation
        sub_opcodes = [inst.opcode for inst in sub_comp.instructions]
        assert "add" in sub_opcodes
        assert sub_opcodes.count("parameter") == 2

    def test_different_data_types(self):
        """Test operations with different data types."""

        def mixed_types(scribe):
            s32 = scribe.s32
            f64 = scribe.f64
            pred = scribe.pred

            x = s32[10].Parameter(parameter_number=0)
            _ = f64[10].Parameter(parameter_number=1)
            _ = pred[10].Parameter(parameter_number=2)

            return s32[10].Add(x, x)  # Just return s32 operation

        scribe = HloScribe()
        result = scribe(mixed_types)

        comp = result.module_proto.computations[0]
        params = [inst for inst in comp.instructions if inst.opcode == "parameter"]

        assert len(params) == 3
        assert params[0].shape.element_type == xla_data_pb2.PrimitiveType.S32
        assert params[1].shape.element_type == xla_data_pb2.PrimitiveType.F64
        assert params[2].shape.element_type == xla_data_pb2.PrimitiveType.PRED

    def test_variadic_operations(self):
        """Test operations that accept variable number of operands."""

        def variadic_func(scribe):
            f32 = scribe.f32
            x = f32[10].Parameter(parameter_number=0)
            y = f32[10].Parameter(parameter_number=1)
            z = f32[10].Parameter(parameter_number=2)

            return f32[30].Concatenate(x, y, z, dimensions=[0])

        scribe = HloScribe()
        result = scribe(variadic_func)

        comp = result.module_proto.computations[0]
        concat_ops = [inst for inst in comp.instructions if inst.opcode == "concatenate"]
        assert len(concat_ops) == 1

        concat_op = concat_ops[0]
        assert len(concat_op.operand_ids) == 3  # Three operands

    def test_tuple_operations(self):
        """Test tuple creation and access."""

        def tuple_func(scribe):
            f32 = scribe.f32
            tuple_ = scribe.tuple

            x = f32[10].Parameter(parameter_number=0)
            y = f32[10].Parameter(parameter_number=1)

            def reducer(scribe):
                f32 = scribe.f32
                p0 = f32.Parameter(parameter_number=0)
                p1 = f32.Parameter(parameter_number=1)
                return f32.Add(p0, p1)

            # Create tuple - this is a simplified test, real implementation might differ
            return tuple_(f32[10], f32[10]).AllReduce(
                x, y, replica_groups=[[0, 1]], to_apply=reducer
            )

        scribe = HloScribe()
        result = scribe(tuple_func)

        # Basic test that it doesn't crash
        assert result.module_proto is not None

    def test_program_shape_setup(self):
        """Test that program shape is correctly set up."""

        def shape_test(scribe):
            f32 = scribe.f32
            bf16 = scribe.bf16

            x = f32[128, 256].Parameter(parameter_number=0)
            _ = bf16[64].Parameter(parameter_number=1)

            return f32[128, 256].Add(x, x)

        scribe = HloScribe()
        result = scribe(shape_test)

        comp = result.module_proto.computations[0]
        prog_shape = comp.program_shape

        # Check parameter shapes
        assert len(prog_shape.parameters) == 2
        assert prog_shape.parameters[0].element_type == xla_data_pb2.PrimitiveType.F32
        assert list(prog_shape.parameters[0].dimensions) == [128, 256]
        assert prog_shape.parameters[1].element_type == xla_data_pb2.PrimitiveType.BF16
        assert list(prog_shape.parameters[1].dimensions) == [64]

        # Check parameter names
        assert prog_shape.parameter_names == ["p0", "p1"]

        # Check result shape
        assert prog_shape.result.element_type == xla_data_pb2.PrimitiveType.F32
        assert list(prog_shape.result.dimensions) == [128, 256]

    def test_error_handling_wrong_arity(self):
        """Test error handling for operations with wrong number of operands."""

        def bad_arity(scribe):
            f32 = scribe.f32
            x = f32[10].Parameter(parameter_number=0)
            # Add expects 2 operands, but we're giving 1
            return f32[10].Add(x)

        scribe = HloScribe()
        with pytest.raises(ValueError, match="Add expects 2 operands"):
            scribe(bad_arity)

    def test_counter_increments(self):
        """Test that instruction IDs are properly incremented."""

        def counter_test(scribe):
            f32 = scribe.f32
            x = f32[10].Parameter(parameter_number=0)
            y = f32[10].Parameter(parameter_number=1)
            z = f32[10].Add(x, y)
            return f32[10].Multiply(z, x)

        scribe = HloScribe()
        result = scribe(counter_test)

        comp = result.module_proto.computations[0]
        ids = [inst.id for inst in comp.instructions]

        # IDs should be unique and incrementing
        assert len(set(ids)) == len(ids)  # All unique
        assert ids == sorted(ids)  # Incrementing

    def test_context_management(self):
        """Test that context is properly managed."""

        def context_test(scribe):
            # During execution, HloScribe.context should be set
            assert HloScribe.context == scribe
            f32 = scribe.f32
            return f32[10].Parameter(parameter_number=0)

        scribe = HloScribe()

        # Before execution, context should be None
        assert HloScribe.context is None

        _ = scribe(context_test)

        # After execution, context should be None again
        assert HloScribe.context is None

    def test_module_serialization(self):
        """Test that modules can be serialized to protobuf."""

        def simple_func(scribe):
            f32 = scribe.f32
            x = f32[10].Parameter(parameter_number=0)
            return f32[10].Abs(x)

        scribe = HloScribe()
        result = scribe(simple_func)

        # Should be able to serialize
        serialized = result.module_proto.SerializeToString()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        # Should be able to deserialize
        new_module = hlo_pb2.HloModuleProto()
        new_module.ParseFromString(serialized)
        assert new_module.name == result.module_proto.name

    def test_file_saving(self):
        """Test saving module to file."""

        def file_test(scribe):
            f32 = scribe.f32
            x = f32[5, 5].Parameter(parameter_number=0)
            return f32[5, 5].Tanh(x)

        scribe = HloScribe()
        result = scribe(file_test)

        # Test saving to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pb", delete=False) as tmp:
            try:
                tmp.write(result.module_proto.SerializeToString())
                tmp.flush()

                # Verify file exists and has content
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0

            finally:
                os.unlink(tmp.name)


class TestProtoSetAttr:
    """Test cases for proto_set_attr utility function."""

    def test_simple_attribute_setting(self):
        """Test setting simple attributes."""
        inst = hlo_pb2.HloInstructionProto()
        proto_set_attr(inst, "parameter_number", 5)
        assert inst.parameter_number == 5

    def test_list_attribute_setting(self):
        """Test setting list attributes."""
        inst = hlo_pb2.HloInstructionProto()
        proto_set_attr(inst, "operand_ids", [1, 2, 3])
        assert list(inst.operand_ids) == [1, 2, 3]

    def test_dict_attribute_setting(self):
        """Test setting nested dictionary attributes."""
        inst = hlo_pb2.HloInstructionProto()
        dot_dims = {"lhs_contracting_dimensions": [0, 1], "rhs_contracting_dimensions": [2, 3]}
        proto_set_attr(inst, "dot_dimension_numbers", dot_dims)

        assert list(inst.dot_dimension_numbers.lhs_contracting_dimensions) == [0, 1]
        assert list(inst.dot_dimension_numbers.rhs_contracting_dimensions) == [2, 3]


class TestHloShape:
    """Test cases for HloShape class."""

    def test_shape_creation(self):
        """Test basic shape creation."""
        scribe = HloScribe()
        shape = HloShape(scribe, xla_data_pb2.PrimitiveType.F32)
        assert shape.shape_proto.element_type == xla_data_pb2.PrimitiveType.F32

    def test_shape_indexing(self):
        """Test shape dimension setting via indexing."""
        scribe = HloScribe()
        f32 = HloShape(scribe, xla_data_pb2.PrimitiveType.F32)
        shaped = f32[10, 20, 30]

        assert list(shaped.shape_proto.dimensions) == [10, 20, 30]
        assert len(shaped.shape_proto.is_dynamic_dimension) == 3
        assert all(not dyn for dyn in shaped.shape_proto.is_dynamic_dimension)

    def test_shape_cloning(self):
        """Test shape cloning."""
        scribe = HloScribe()
        original = HloShape(scribe, xla_data_pb2.PrimitiveType.S32)
        original.shape_proto.dimensions[:] = [5, 10]

        cloned = original.clone()
        assert cloned.shape_proto.element_type == original.shape_proto.element_type
        assert list(cloned.shape_proto.dimensions) == [5, 10]

        # Verify they're independent
        cloned.shape_proto.dimensions[0] = 15
        assert original.shape_proto.dimensions[0] == 5


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_function(self):
        """Test function that returns None."""

        def empty_func(scribe):
            pass  # Returns None

        scribe = HloScribe()
        with pytest.raises(AttributeError):
            scribe(empty_func)

    def test_start_counter(self):
        """Test HloScribe with custom start counter."""
        scribe = HloScribe(start_counter=100)

        def simple(scribe):
            f32 = scribe.f32
            return f32[10].Parameter(parameter_number=0)

        result = scribe(simple)
        comp = result.module_proto.computations[0]

        # First instruction should have ID > 100
        assert comp.instructions[0].id > 100

    def test_complex_nested_operations(self):
        """Test complex nested operations."""

        def complex_func(scribe):
            f32 = scribe.f32

            # Create multiple parameters
            inputs = []
            for i in range(5):
                inputs.append(f32[10].Parameter(parameter_number=i))

            # Chain operations
            result = inputs[0]
            for i in range(1, 5):
                if i % 2 == 0:
                    result = f32[10].Add(result, inputs[i])
                else:
                    result = f32[10].Multiply(result, inputs[i])

            return f32[10].Tanh(result)

        scribe = HloScribe()
        result = scribe(complex_func)

        comp = result.module_proto.computations[0]

        # Should have 5 parameters + 4 binary ops + 1 unary op = 10 instructions
        assert len(comp.instructions) == 10

        # Check we have the right number of each operation type
        opcodes = [inst.opcode for inst in comp.instructions]
        assert opcodes.count("parameter") == 5
        assert opcodes.count("add") == 2  # Even indices
        assert opcodes.count("multiply") == 2  # Odd indices
        assert opcodes.count("tanh") == 1


# Integration test
def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""

    def neural_network_layer(scribe):
        """Simulate a simple neural network layer: W*x + b with activation."""
        f32 = scribe.f32

        # Inputs
        x = f32[128, 512].Parameter(parameter_number=0)  # Input
        w = f32[512, 256].Parameter(parameter_number=1)  # Weight matrix
        b = f32[256].Parameter(parameter_number=2)  # Bias

        # Matrix multiplication
        dot_dims = {"lhs_contracting_dimensions": [1], "rhs_contracting_dimensions": [0]}
        matmul = f32[128, 256].Dot(x, w, dot_dimension_numbers=dot_dims)

        # Add bias (broadcast)
        bias_bc = f32[128, 256].Broadcast(b, dimensions=[1])
        linear = f32[128, 256].Add(matmul, bias_bc)

        # Activation
        return f32[128, 256].Tanh(linear)

    # Create and compile
    scribe = HloScribe()
    result = scribe(neural_network_layer)

    # Verify structure
    assert result.module_proto.name == "neural_network_layer.8"

    comp = result.module_proto.computations[0]
    assert len(comp.program_shape.parameters) == 3

    # Verify operations are present
    opcodes = [inst.opcode for inst in comp.instructions]
    required_ops = ["parameter", "dot", "broadcast", "add", "tanh"]
    for op in required_ops:
        assert op in opcodes

    # Verify can serialize
    serialized = result.module_proto.SerializeToString()
    assert len(serialized) > 0

    print("End-to-end test passed!")
