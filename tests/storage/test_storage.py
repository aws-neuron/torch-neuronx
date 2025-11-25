import torch

import torch_neuronx


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

    def test_cpu_to_neuron_storage_conversion(self):
        """Test converting CPU storage to neuron device."""
        cpu_tensor = torch.randn(3, 4)
        cpu_storage = cpu_tensor.untyped_storage()

        # Convert storage to neuron
        neuron_storage = cpu_storage.to(device="neuron")

        assert neuron_storage.device.type == "neuron"
        assert neuron_storage.nbytes() == cpu_storage.nbytes()

    def test_empty_storage_conversion(self):
        """Test converting empty storage to neuron."""
        empty_storage = torch.UntypedStorage(0)
        neuron_storage = empty_storage.to(device="neuron")

        assert neuron_storage.device.type == "neuron"
        assert neuron_storage.nbytes() == 0

    def test_storage_data_integrity(self):
        """Test that data is preserved during storage conversion."""
        # Create tensor with known values
        cpu_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        original_data = cpu_tensor.clone()

        # Convert storage and create new tensor from it
        cpu_storage = cpu_tensor.untyped_storage()
        neuron_storage = cpu_storage.to(device="neuron")

        # Create tensor from converted storage
        neuron_tensor = (
            torch.tensor([], dtype=torch.float32, device="neuron")
            .set_(neuron_storage)
            .reshape(3, 4)
        )

        # Verify data integrity by moving back to CPU
        recovered_tensor = neuron_tensor.cpu()
        torch.testing.assert_close(recovered_tensor, original_data)

    def test_storage_copy_operations(self):
        """Test storage copy operations that are used internally by torch.load()"""
        # Create a neuron tensor and get its storage
        neuron_tensor = torch.randn(5, 3, device="neuron")
        neuron_storage = neuron_tensor.untyped_storage()

        # Create a new empty storage on the same device
        new_storage = torch.UntypedStorage(neuron_storage.nbytes(), device=neuron_storage.device)

        # This is the exact line that segfaults in PyTorch's storage.py:261
        new_storage.copy_(neuron_storage)

        # Verify the copy worked by creating tensors from both storages
        original_tensor = (
            torch.tensor([], dtype=torch.float32, device="neuron")
            .set_(neuron_storage)
            .reshape(5, 3)
        )
        copied_tensor = (
            torch.tensor([], dtype=torch.float32, device="neuron").set_(new_storage).reshape(5, 3)
        )

        torch.testing.assert_close(original_tensor.cpu(), copied_tensor.cpu())

    def test_storage_creation_with_device(self):
        """Test creating UntypedStorage directly with neuron device"""
        # Test different sizes
        for size in [0, 100, 1000]:
            storage = torch.UntypedStorage(size, device="neuron")
            assert storage.device.type == "neuron"
            assert storage.nbytes() == size

    def test_storage_resize_operations(self):
        """Test storage resize operations"""
        storage = torch.UntypedStorage(100, device="neuron")

        # Test resize larger
        storage.resize_(200)
        assert storage.nbytes() == 200

        # Test resize smaller
        storage.resize_(50)
        assert storage.nbytes() == 50

    def test_serialization_location_tag(self):
        """Test serialization location tag for neuron storage (like openreg test)"""
        storage = torch.UntypedStorage(4, device=torch.device("neuron"))
        assert torch.serialization.location_tag(storage) == "neuron:0"

        storage = torch.UntypedStorage(4, device=torch.device("neuron:0"))
        assert torch.serialization.location_tag(storage) == "neuron:0"

    def test_storage_restore_location(self):
        """Test storage restore location for serialization"""
        storage_cpu = torch.empty(4, 4).storage()
        storage_neuron = torch.serialization.default_restore_location(storage_cpu, "neuron:0")
        assert storage_neuron.device.type == "neuron"

    def test_rewrapped_storage(self):
        """Test storage rewrapping (from openreg test pattern)"""
        # Create a neuron tensor
        neuron_tensor = torch.randn(10, device="neuron")

        # Create a rewrapped tensor using storage slicing
        rewrapped_tensor = torch.tensor((), dtype=torch.float32, device="neuron").set_(
            neuron_tensor.untyped_storage()[2:],
            size=(5,),
            stride=(1,),
            storage_offset=0,
        )

        assert rewrapped_tensor.device.type == "neuron"
        assert rewrapped_tensor.numel() == 5
        # Data pointers should be different (offset)
        assert neuron_tensor.data_ptr() != rewrapped_tensor.data_ptr()
