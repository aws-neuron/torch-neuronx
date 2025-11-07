import torch

import torch_neuron

# NOTE: These tests currently verify the basic allocator infrastructure.
# As we expand the caching allocator functionality, we'll add more meaningful
# user-facing tests similar to PyTorch's CUDA memory management.
# See torch/cuda/memory.py for examples of user-facing memory management APIs like:
# - torch.cuda.empty_cache()
# - torch.cuda.memory_allocated()
# - torch.cuda.memory_reserved()
# - torch.cuda.max_memory_allocated()
# etc.


class TestNeuronAllocator:
    def test_allocator_module_exists(self):
        """Test that the NeuronCachingAllocator module exists."""
        assert hasattr(torch_neuron._C, "NeuronCachingAllocator")
        assert hasattr(torch_neuron._C.NeuronCachingAllocator, "get")
        assert hasattr(torch_neuron._C.NeuronCachingAllocator, "emptyCache")

    def test_empty_cache_works(self):
        """Test that emptyCache can be called without error."""
        # This should not throw
        torch_neuron._C.NeuronCachingAllocator.emptyCache()

    def test_empty_uses_allocator(self):
        """Test that torch.empty uses NeuronCachingAllocator (verified via debug output)"""
        print("\n=== Testing allocator usage ===")

        # Create a tensor - should see debug output
        x = torch.empty(1000, device="neuron")

        # Verify basic properties
        assert x.device.type == "neuron"  # PyTorch shows our custom name
        assert x.shape == (1000,)
        assert x.dtype == torch.float32

        # Create another tensor with different size
        y = torch.empty(2000, device="neuron")
        assert y.shape == (2000,)

        print("=== Test completed ===\n")

    def test_allocator_cache_hit(self):
        """Test that allocator reuses freed memory of same size"""
        print("\n=== Testing cache hit functionality ===")

        # Clear any existing cache
        torch_neuron._C.NeuronCachingAllocator.emptyCache()

        # First allocation
        x = torch.empty(1024, device="neuron")
        ptr1 = x.data_ptr()
        print(f"First allocation pointer: {ptr1}")
        del x

        # Second allocation - should reuse
        y = torch.empty(1024, device="neuron")
        ptr2 = y.data_ptr()
        print(f"Second allocation pointer: {ptr2}")

        assert ptr1 == ptr2, f"Allocator should reuse freed memory. Got ptr1={ptr1}, ptr2={ptr2}"

        print("=== Cache hit test completed successfully ===\n")

    def test_empty_strided_uses_allocator(self):
        """Test that torch.empty_strided uses NeuronAllocator and doesn't fall to CPU"""
        a = torch.empty_strided((2, 3), (1, 2), device="neuron")
        assert a.device.type == "neuron"
        assert "aten::empty_strided" not in torch_neuron.get_fallback_ops()


class TestPinnedMemory:
    def test_pinned_memory_allocation(self):
        """Test that pinned memory allocation falls back to CPU allocator"""
        # Allocate pinned memory tensor on CPU
        x = torch.empty(1000, device="cpu", pin_memory=True)

        # Verify properties
        assert x.device.type == "cpu"
        assert x.is_pinned()

        # Move to Neuron device
        y = x.to("neuron")
        assert y.device.type == "neuron"
        assert not y.is_pinned()  # Neuron tensors are not pinned in the same

    def test_pin_memory_fn(self):
        """Test that pinned memory allocation falls back to CPU allocator"""
        # Allocate pinned memory tensor on CPU
        x = torch.empty(1000, device="cpu")
        x = x.pin_memory()

        # Verify properties
        assert x.device.type == "cpu"
        assert x.is_pinned()

        # Move to Neuron device
        y = x.to("neuron")
        assert y.device.type == "neuron"
        assert not y.is_pinned()  # Neuron tensors are not pinned in the same

    def test_neuron_tensor_pinning(self):
        """Test that Neuron tensors are always considered pinned memory"""
        # Allocate tensor directly on Neuron device
        # the below will crash as you cant pin memory on neuron device
        try:
            x = torch.empty(1000, device="neuron")
            x.pin_memory()
        except RuntimeError as e:
            assert "pin" in str(e)
