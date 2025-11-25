import pytest
import torch

from torch_neuronx.python_ops.xla_builder.type_converter import XLABuilderTypeConverter


class TestAllGatherXlaOpLowering:
    """Unit tests for AllGatherXlaOp HLO lowering."""

    @pytest.fixture
    def all_gather_op(self):
        """Fixture to create AllGatherXlaOp instance."""
        from torch_neuronx.python_ops.xla_ops.all_gather_xla import AllGatherXlaOp

        return AllGatherXlaOp()

    def test_basic_lowering(self, all_gather_op):
        """Test basic HLO lowering with single input tensor."""
        # Create input tensor
        input_tensor = torch.ones(10, dtype=torch.float32)
        replica_groups = [[0, 1]]  # Two replicas

        # Generate HLO
        hlo_module = all_gather_op.hlo_fn(
            [input_tensor], replica_groups=replica_groups, slice_output=False
        )

        # Verify HLO structure
        assert hlo_module.module_proto is not None
        computation = hlo_module.module_proto.computations[0]

        # Verify the presence of AllGather operation
        all_gather_ops = [inst for inst in computation.instructions if inst.opcode == "all-gather"]
        assert len(all_gather_ops) == 1

        # Verify AllGather configuration
        all_gather_op = all_gather_ops[0]
        assert all_gather_op.dimensions == [0]  # Gathering along dimension 0
        assert len(all_gather_op.replica_groups) == 1
        assert all_gather_op.replica_groups[0].replica_ids == [0, 1]

    def test_multi_input_lowering(self, all_gather_op):
        """Test HLO lowering with multiple input tensors."""
        input_tensors = [torch.ones(10, dtype=torch.float32), torch.ones(10, dtype=torch.float32)]
        replica_groups = [[0, 1]]

        hlo_module = all_gather_op.hlo_fn(
            input_tensors, replica_groups=replica_groups, slice_output=False
        )

        computation = hlo_module.module_proto.computations[0]

        # Verify parameter count
        parameters = [inst for inst in computation.instructions if inst.opcode == "parameter"]
        assert len(parameters) == 2

        # Verify output shape
        root = next(inst for inst in computation.instructions if inst.id == computation.root_id)
        assert root.shape.tuple_shapes[0].dimensions[0] == 20  # 2 * input size

    def test_sliced_output_lowering(self, all_gather_op):
        """Test HLO lowering with sliced output."""
        input_tensor = torch.ones(10, dtype=torch.float32)
        replica_groups = [[0, 1]]

        hlo_module = all_gather_op.hlo_fn(
            [input_tensor], replica_groups=replica_groups, slice_output=True
        )

        computation = hlo_module.module_proto.computations[0]

        # Verify presence of Slice operations
        slice_ops = [inst for inst in computation.instructions if inst.opcode == "slice"]
        assert len(slice_ops) == 2  # One slice per replica

        # Verify final Tuple operation for multiple outputs
        root = next(inst for inst in computation.instructions if inst.id == computation.root_id)
        assert root.opcode == "tuple"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_lowering(self, all_gather_op, dtype):
        """Test HLO lowering with different dtypes."""
        input_tensor = torch.ones(10, dtype=dtype)
        replica_groups = [[0, 1]]

        hlo_module = all_gather_op.hlo_fn(
            [input_tensor], replica_groups=replica_groups, slice_output=False
        )

        computation = hlo_module.module_proto.computations[0]

        # Verify correct dtype in HLO
        parameter = next(inst for inst in computation.instructions if inst.opcode == "parameter")
        expected_primitive_type = XLABuilderTypeConverter.torch_to_primitive_dtype(dtype)
        assert parameter.shape.element_type == expected_primitive_type

    @pytest.mark.parametrize("shape", [(10,), (5, 2), (2, 3, 4)])
    def test_shape_lowering(self, all_gather_op, shape):
        """Test HLO lowering with different input shapes."""
        input_tensor = torch.ones(shape, dtype=torch.float32)
        replica_groups = [[0, 1]]

        hlo_module = all_gather_op.hlo_fn(
            [input_tensor], replica_groups=replica_groups, slice_output=False
        )

        computation = hlo_module.module_proto.computations[0]

        # Verify output shape
        root = next(inst for inst in computation.instructions if inst.id == computation.root_id)
        expected_shape = list(shape)
        expected_shape[0] *= 2  # First dimension doubled for 2 replicas
        assert list(root.shape.dimensions) == expected_shape

    def test_replica_groups_lowering(self, all_gather_op):
        """Test HLO lowering with different replica group configurations."""
        input_tensor = torch.ones(10, dtype=torch.float32)
        replica_groups = [[0, 1, 2, 3]]  # Four replicas

        hlo_module = all_gather_op.hlo_fn(
            [input_tensor], replica_groups=replica_groups, slice_output=False
        )

        computation = hlo_module.module_proto.computations[0]

        all_gather_op = next(
            inst for inst in computation.instructions if inst.opcode == "all-gather"
        )
        assert len(all_gather_op.replica_groups) == 1
        assert all_gather_op.replica_groups[0].replica_ids == [0, 1, 2, 3]
