import torch

import torch_neuron


class TestNeuronStorage:
    def test_empty_tensor_creates_storage(self):
        """Test that torch.empty creates a tensor with neuron storage"""
        # The simplest test - create an empty tensor on neuron device
        tensor = torch.empty(10, device="neuron")

        # Verify the tensor is on neuron device
        assert tensor.device.type == "neuron"
        assert tensor.device.index == 0

        # Verify the tensor has correct properties
        assert tensor.shape == (10,)
        assert tensor.numel() == 10
        assert tensor.dtype == torch.float32

        # Verify the underlying storage
        storage = tensor.untyped_storage()
        assert storage.device.type == "neuron"
        assert storage.size() == 10 * tensor.element_size()

    def test_empty_tensor_different_shapes(self):
        """Test creating tensors with different shapes"""
        # 1D tensor
        t1 = torch.empty(100, device="neuron")
        assert t1.shape == (100,)

        # 2D tensor
        t2 = torch.empty(10, 20, device="neuron")
        assert t2.shape == (10, 20)
        assert t2.numel() == 200

        # 3D tensor
        t3 = torch.empty(2, 3, 4, device="neuron")
        assert t3.shape == (2, 3, 4)
        assert t3.numel() == 24

    def test_empty_tensor_different_dtypes(self):
        """Test creating tensors with different data types"""
        # Float tensor (default)
        t_float = torch.empty(10, dtype=torch.float32, device="neuron")
        assert t_float.dtype == torch.float32
        assert t_float.element_size() == 4

        # Double tensor
        t_double = torch.empty(10, dtype=torch.float64, device="neuron")
        assert t_double.dtype == torch.float64
        assert t_double.element_size() == 8

        # Int tensor
        t_int = torch.empty(10, dtype=torch.int32, device="neuron")
        assert t_int.dtype == torch.int32
        assert t_int.element_size() == 4
