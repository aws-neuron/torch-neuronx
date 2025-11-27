import pytest
import torch

# Skip these tests if JAX isn't available
pytest.importorskip("jax")

from torch_neuronx.python_ops.jax.handlers.output import OutputHandler
from torch_neuronx.python_ops.jax.type_converter import JaxTypeConverter


def to_torch_dtype(spec_dtype) -> torch.dtype:
    """Helper to convert JAX/NumPy dtype from ShapeDtypeStruct to torch dtype."""
    return JaxTypeConverter.jax_to_torch_dtype(spec_dtype)


class TestMetaInference:
    def test_ones_no_tensor_args_default_dtype_device_neuron(self):
        """
        Meta inference should work with no tensor inputs and device='neuron:0'.
        Expected dtype uses PyTorch default dtype.
        """
        handler = OutputHandler()

        size = (2, 3)
        specs, is_single, expected_dtypes, _ = handler.infer_output_specs(
            jax_fn=lambda *a, **k: None,
            inputs=(size,),
            op_name="aten::ones",
            static_argnums=(0,),
            kwargs={"device": "neuron:0"},
        )

        assert is_single is True
        assert len(specs) == 1
        assert expected_dtypes == [torch.get_default_dtype()]
        # Shape matches requested size
        assert tuple(specs[0].shape) == size
        # Spec dtype corresponds to default torch dtype
        assert to_torch_dtype(specs[0].dtype) == torch.get_default_dtype()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32, torch.bool])
    def test_ones_explicit_dtype_respected(self, dtype):
        """
        When dtype is provided, meta inference should respect it for expected dtypes
        and produce matching JAX spec dtype.
        """
        handler = OutputHandler()

        size = (3, 2)
        specs, is_single, expected_dtypes, _ = handler.infer_output_specs(
            jax_fn=lambda *a, **k: None,
            inputs=(size,),
            op_name="aten::ones",
            static_argnums=(0,),
            kwargs={"dtype": dtype, "device": "neuron:0"},
        )

        assert is_single is True
        assert expected_dtypes == [dtype]
        assert tuple(specs[0].shape) == size
        assert to_torch_dtype(specs[0].dtype) == dtype

    def test_ones_out_param_shape_mismatch_respects_size_and_out_dtype(self):
        """
        With out provided, meta inference should:
        - Use the requested size for the output shape (not current out.shape)
        - Use out.dtype as the expected PyTorch dtype when dtype kwarg is absent
        """
        handler = OutputHandler()

        size = (2, 3)
        out = torch.empty((1,), dtype=torch.float16)  # wrong shape on purpose

        specs, is_single, expected_dtypes, _ = handler.infer_output_specs(
            jax_fn=lambda *a, **k: None,
            inputs=(size,),
            op_name="aten::ones.out",
            static_argnums=(0,),
            kwargs={"out": out},
        )

        assert is_single is True
        assert expected_dtypes == [torch.float16]
        # Shape must match requested size, not out's original shape
        assert tuple(specs[0].shape) == size
        assert to_torch_dtype(specs[0].dtype) == torch.float16

    def test_max_dim_indices_int64_mapped_to_int32_in_specs(self):
        """
        For ops producing index outputs (int64 in PyTorch), meta should report
        expected_dtypes with int64, while the JAX execution spec maps to int32.
        """
        handler = OutputHandler()

        x = torch.randn(4, 5, dtype=torch.float32)
        # aten::max.dim returns (values, indices)
        specs, is_single, expected_dtypes, _ = handler.infer_output_specs(
            jax_fn=lambda *a, **k: None,
            inputs=(x,),
            op_name="aten::max.dim",
            static_argnums=(1, 2),
            kwargs={"dim": 1, "keepdim": False},
        )

        assert is_single is False
        assert len(specs) == 2
        # Expected PyTorch dtypes: values=float32, indices=int64
        assert expected_dtypes[0] == torch.float32
        assert expected_dtypes[1] == torch.int64

        # Spec dtypes used for JAX execution: second output should be int32
        assert to_torch_dtype(specs[0].dtype) == torch.float32
        assert to_torch_dtype(specs[1].dtype) == torch.int32

    def test_expected_dtypes_override(self):
        """
        expected_dtypes_override should replace meta-inferred PyTorch dtypes while
        still mapping JAX spec dtype appropriately.
        """
        handler = OutputHandler()

        size = (2, 2)
        override_dtype = torch.int32
        specs, is_single, expected_dtypes, _ = handler.infer_output_specs(
            jax_fn=lambda *a, **k: None,
            inputs=(size,),
            op_name="aten::ones",
            static_argnums=(0,),
            expected_dtypes_override=[override_dtype],
            kwargs={"device": "neuron:0"},
        )

        assert is_single is True
        assert expected_dtypes == [override_dtype]
        assert tuple(specs[0].shape) == size
        assert to_torch_dtype(specs[0].dtype) == override_dtype

    def test_overload_resolution_max_vs_max_dim(self):
        """
        Meta inference should resolve the correct ATen overload:
        - aten::max (unary) returns a scalar (single output)
        - aten::max.dim returns (values, indices)
        """
        handler = OutputHandler()
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        # Unary max (reduce all dims)
        specs1, is_single1, dtypes1, _ = handler.infer_output_specs(
            jax_fn=lambda *a, **k: None,
            inputs=(x,),
            op_name="aten::max",
            kwargs={},
        )
        assert is_single1 is True
        assert len(specs1) == 1
        # Scalar output has empty shape
        assert tuple(specs1[0].shape) == ()
        assert dtypes1 == [torch.float32]

        # max.dim variant
        specs2, is_single2, dtypes2, _ = handler.infer_output_specs(
            jax_fn=lambda *a, **k: None,
            inputs=(x,),
            op_name="aten::max.dim",
            static_argnums=(1, 2),
            kwargs={"dim": 1, "keepdim": False},
        )
        assert is_single2 is False
        assert len(specs2) == 2
        # Values dtype float32; indices int64 in PyTorch expectations
        assert dtypes2[0] == torch.float32
        assert dtypes2[1] == torch.int64
        # JAX specs: indices mapped to int32
        assert to_torch_dtype(specs2[0].dtype) == torch.float32
        assert to_torch_dtype(specs2[1].dtype) == torch.int32
