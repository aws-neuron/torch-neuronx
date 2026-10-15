import torch

import torch_neuronx

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
        assert hasattr(torch_neuronx._C, "NeuronCachingAllocator")
        assert hasattr(torch_neuronx._C.NeuronCachingAllocator, "get")
        assert hasattr(torch_neuronx._C.NeuronCachingAllocator, "emptyCache")

    def test_empty_cache_works(self):
        """Test that emptyCache can be called without error."""
        # This should not throw
        torch_neuronx._C.NeuronCachingAllocator.emptyCache()

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

    def test_storage_uses_allocator(self):
        """Test that storage creation uses our allocator"""
        # Create storage directly - should use our allocator
        storage = torch.UntypedStorage(1000, device="neuron")
        assert storage.device.type == "neuron"
        assert storage.nbytes() == 1000

    def test_empty_strided_uses_allocator(self):
        """Test that torch.empty_strided uses NeuronAllocator and doesn't fall to CPU"""
        a = torch.empty_strided((2, 3), (1, 2), device="neuron")
        assert a.device.type == "neuron"
        assert "aten::empty_strided" not in torch_neuronx.get_fallback_ops()

    def test_empty_strided_non_contiguous_storage_size(self):
        """Storage size must cover all elements reachable via non-contiguous strides."""
        shape = (1, 1, 512, 384)
        strides = (589824, 589824, 1152, 1)
        dtype = torch.bfloat16

        cpu = torch.empty_strided(shape, strides, dtype=dtype, device="cpu")
        neu = torch.empty_strided(shape, strides, dtype=dtype, device="neuron")

        assert neu.untyped_storage().size() == cpu.untyped_storage().size(), (
            f"storage size mismatch: neuron={neu.untyped_storage().size()} "
            f"cpu={cpu.untyped_storage().size()}"
        )

        # copy_ must succeed (previously failed with out-of-bounds storage error)
        src = torch.randn(shape, dtype=dtype, device="neuron")
        neu.copy_(src)
        assert torch.allclose(neu.cpu().float(), src.cpu().float())


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

    def test_unpinned_cpu_tensor_is_not_pinned(self):
        """Test that regular CPU tensors are NOT reported as pinned.
        This is the core fix: previously isPinnedPtr always returned true."""
        x = torch.empty(1000, device="cpu")
        assert not x.is_pinned(), "Regular CPU tensor should not be pinned"

    def test_neuron_to_cpu_is_not_pinned(self):
        """Test that a tensor copied from Neuron to CPU is not pinned"""
        x = torch.empty(1000, device="neuron")
        y = x.to("cpu")
        assert not y.is_pinned(), "Tensor moved from Neuron to CPU should not be pinned"

    def test_multiple_pinned_allocations(self):
        """Test that multiple concurrent pinned allocations are all tracked correctly"""
        tensors = [torch.empty(100 * (i + 1), pin_memory=True) for i in range(5)]
        for i, t in enumerate(tensors):
            assert t.is_pinned(), f"Pinned tensor {i} should be pinned"

    def test_pinned_memory_after_del(self):
        """Test that is_pinned is correct after some pinned tensors are freed"""
        a = torch.empty(1000, pin_memory=True)
        b = torch.empty(2000, pin_memory=True)
        assert a.is_pinned()
        assert b.is_pinned()
        del a
        assert b.is_pinned(), "Remaining pinned tensor should still be pinned"

    def test_data_ptr_returns_nonzero(self):
        """data_ptr() on a neuron tensor should return a nonzero mmap VA."""
        x = torch.empty(64, device="neuron")
        assert x.data_ptr() != 0

    def test_data_ptr_is_stable(self):
        """Calling data_ptr() twice on the same tensor should return the same address."""
        x = torch.empty(64, device="neuron")
        assert x.data_ptr() == x.data_ptr()

    def test_different_tensors_different_ptrs(self):
        """Two distinct tensors should have different data_ptr values."""
        a = torch.empty(64, device="neuron")
        b = torch.empty(64, device="neuron")
        assert a.data_ptr() != b.data_ptr()

    def test_data_ptr_after_operation(self):
        """The result of an operation should have a valid (nonzero) data_ptr."""
        a = torch.ones(64, device="neuron")
        b = torch.ones(64, device="neuron")
        c = a + b
        assert c.data_ptr() != 0

    def test_data_ptr_various_sizes(self):
        """data_ptr() should be nonzero for small, medium, and large tensors."""
        for size in [1, 1024, 1_000_000]:
            x = torch.empty(size, device="neuron")
            assert x.data_ptr() != 0, f"data_ptr was zero for size {size}"

    def test_data_ptr_after_del(self):
        """After deleting a tensor, a newly created tensor should still get a valid data_ptr."""
        x = torch.empty(64, device="neuron")
        _ = x.data_ptr()
        del x
        y = torch.empty(64, device="neuron")
        assert y.data_ptr() != 0

    def test_data_ptr_with_empty_cache(self):
        """data_ptr should remain valid after empty_cache is called."""
        x = torch.empty(64, device="neuron")
        ptr_before = x.data_ptr()
        torch_neuronx._C.NeuronCachingAllocator.emptyCache()
        assert x.data_ptr() == ptr_before
