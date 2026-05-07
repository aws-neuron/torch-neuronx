import copy
import os
import tempfile

import pytest
import torch

import torch_neuronx


class TestTorchSaveLoad:
    """Test torch.load with Neuron tensors."""

    def test_save_load_single_tensor(self):
        """Test saving and loading a single neuron tensor."""
        original = torch.randn(5, 3, device="neuron")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(original, f.name)
            loaded = torch.load(f.name)
            os.unlink(f.name)

        assert loaded.device.type == "neuron"
        assert loaded.shape == original.shape
        torch.testing.assert_close(loaded.cpu(), original.cpu())

    def test_save_load_tensor_dict(self):
        """Test saving and loading dictionary of neuron tensors."""
        tensors = {
            "weight": torch.randn(10, 5, device="neuron"),
            "bias": torch.randn(10, device="neuron"),
            "scale": torch.tensor(2.5, device="neuron"),
        }

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(tensors, f.name)
            loaded = torch.load(f.name)
            os.unlink(f.name)

        assert set(loaded.keys()) == set(tensors.keys())
        for key in tensors:
            assert loaded[key].device.type == "neuron"
            torch.testing.assert_close(loaded[key].cpu(), tensors[key].cpu())

    def test_save_load_mixed_devices(self):
        """Test saving and loading tensors from mixed devices."""
        mixed_tensors = {
            "cpu_tensor": torch.randn(3, 3),
            "neuron_tensor": torch.randn(3, 3, device="neuron"),
        }

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(mixed_tensors, f.name)
            loaded = torch.load(f.name)
            os.unlink(f.name)

        assert loaded["cpu_tensor"].device.type == "cpu"
        assert loaded["neuron_tensor"].device.type == "neuron"
        torch.testing.assert_close(loaded["cpu_tensor"], mixed_tensors["cpu_tensor"])
        torch.testing.assert_close(
            loaded["neuron_tensor"].cpu(), mixed_tensors["neuron_tensor"].cpu()
        )

    def test_save_load_with_map_location(self):
        """Test loading neuron tensor with map_location to CPU."""
        neuron_tensor = torch.randn(3, 4, device="neuron")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(neuron_tensor, f.name)
            # Load to CPU using map_location
            loaded_cpu = torch.load(f.name, map_location="cpu")
            os.unlink(f.name)

        assert loaded_cpu.device.type == "cpu"
        torch.testing.assert_close(loaded_cpu, neuron_tensor.cpu())

    def test_save_load_zero_size_tensor(self):
        """Test saving and loading zero-size neuron tensor."""
        empty_tensor = torch.empty(0, 5, device="neuron")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(empty_tensor, f.name)
            loaded = torch.load(f.name)
            os.unlink(f.name)

        assert loaded.device.type == "neuron"
        assert loaded.shape == (0, 5)

    def test_multiple_save_load_cycles(self):
        """Test multiple save/load cycles preserve data integrity."""
        original = torch.randn(4, 6, device="neuron")
        current = original.clone()

        # Perform multiple save/load cycles
        for _ in range(3):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                torch.save(current, f.name)
                current = torch.load(f.name)
                os.unlink(f.name)

        assert current.device.type == "neuron"
        torch.testing.assert_close(current.cpu(), original.cpu())


class TestDeepCopy:
    """Test deepcopy operations with Neuron tensors."""

    def test_basic_deepcopy(self):
        """Test basic deepcopy of neuron tensors"""
        x = torch.randn(5, 3, device="neuron")
        x_copy = copy.deepcopy(x)
        assert x_copy.device.type == "neuron"
        assert x_copy.data_ptr() != x.data_ptr()
        torch.testing.assert_close(x_copy.cpu(), x.cpu())

    def test_dict_deepcopy(self):
        """Test deepcopy of dict with neuron tensors"""
        tensors = {
            "weight": torch.randn(10, 5, device="neuron"),
            "bias": torch.randn(10, device="neuron"),
        }
        copied = copy.deepcopy(tensors)

        for key in tensors:
            assert copied[key].device.type == "neuron"
            assert copied[key].data_ptr() != tensors[key].data_ptr()
            torch.testing.assert_close(copied[key].cpu(), tensors[key].cpu())

    def test_optimizer_state_deepcopy(self):
        """Test deepcopy of optimizer state dict (original FSDP failing case)"""
        model = torch.nn.Linear(5, 3).to("neuron")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Populate optimizer state
        x = torch.randn(10, 5, device="neuron")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # This was the failing operation
        state_dict = optimizer.state_dict()
        copied_state_dict = copy.deepcopy(state_dict)

        assert copied_state_dict is not None
