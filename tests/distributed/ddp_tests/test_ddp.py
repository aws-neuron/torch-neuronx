from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

try:
    import torch_neuronx
except ImportError:
    torch_neuronx = None

from ..collective_ops.base_collective_op import BaseCollectiveOpTest
from .ddp_test_framework import DDPTestOrchestrator
from .utils import compare_ddp_buckets, compute_ddp_buckets_from_ddp


class SimpleNet(nn.Module):
    """A simple neural network with two linear layers and a ReLU activation.

    Args:
        input_size (int): The number of input features. Defaults to 10.
        hidden_size (int): The number of features in the hidden layer. Defaults to 20.
        output_size (int): The number of output features. Defaults to 5.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleNetWithNoBias(nn.Module):
    """A simple neural network with two linear layers and a ReLU activation, with no bias terms.

    Args:
        input_size (int): The number of input features. Defaults to 10.
        hidden_size (int): The number of features in the hidden layer. Defaults to 20.
        output_size (int): The number of output features. Defaults to 5.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ModelWithUnusedParams(nn.Module):
    """A neural network model containing both used and intentionally unused linear layers.

    This model is designed to test DDP's `find_unused_parameters` functionality.

    Args:
        input_size (int): The number of input features. Defaults to 10.
    """

    def __init__(self, input_size: int = 10):
        super().__init__()
        self.used_layer = nn.Linear(input_size, 5, bias=False)
        self.unused_layer = nn.Linear(5, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.used_layer(x)


class MultiParamModel(nn.Module):
    """A model with multiple parameter groups of varying sizes for bucket capacity testing.

    This model contains layers with different parameter counts to test how DDP
    organizes parameters into buckets based on the bucket capacity limit.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)  # ~0.4KB
        self.fc2 = nn.Linear(10, 10)  # ~0.4KB
        self.fc3 = nn.Linear(10, 100)  # ~4.4KB
        self.fc4 = nn.Linear(100, 1000)  # ~404KB

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def run_bucket_cap_validation(rank: int, world_size: int, kwargs: dict[str, Any]) -> None:
    """Test function for DDP bucket capacity with multiple parameter groups.

    Args:
        rank (int): Process rank in distributed setup.
        world_size (int): Total number of processes.
        kwargs (dict[str, Any]): Additional arguments including bucket_cap_mb.
    """
    bucket_cap_mb = kwargs.get("bucket_cap_mb", 0.05)
    if torch_neuronx is None:
        raise ImportError("torch_neuronx is required for this test")

    device = f"neuron:{torch_neuronx.current_device()}"
    model = MultiParamModel().to(device)
    ddp_model = DistributedDataParallel(model, bucket_cap_mb=bucket_cap_mb, init_sync=False)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for _ in range(5):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(8, 10).to(device))
        labels = torch.randn(8, 1000).to(device)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    # Verify bucket configuration
    if rank == 0:
        ddp_logging_data = ddp_model._get_ddp_logging_data()
        buckets, bucket_sizes = compute_ddp_buckets_from_ddp(
            ddp_model, int(bucket_cap_mb * 1024 * 1024)
        )
        compare_ddp_buckets(ddp_logging_data, buckets, bucket_sizes)
    torch.distributed.barrier()


class TestDDPSimple(BaseCollectiveOpTest):
    """Comprehensive DDP comparison tests using the modular framework.

    This test suite compares the behavior of DistributedDataParallel (DDP) on CPU
    against Neuron devices for various configurations and model types.
    """

    def setup_method(self):
        """Set up the test environment before each test method.

        Initializes `DDPTestOrchestrator` for overall test management.
        """
        self.orchestrator = DDPTestOrchestrator(world_size=2)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_ddp_different_dtypes(self, dtype: torch.dtype):
        """Tests DDP functionality with different data types (float32 and bfloat16).

        Args:
            dtype (torch.dtype): The data type to use for the model and tensors.
        """
        self.orchestrator.run_comparison_test(
            model_class=SimpleNet,
            neuron_tester=self.distributed_tester,
            batch_size=8,
            dtype=dtype,
            test_name=f"ddp_dtype_{dtype}",
        )

    def test_ddp_with_no_bias(self):
        """Tests DDP functionality with a model where linear layers have no bias."""
        self.orchestrator.run_comparison_test(
            model_class=SimpleNetWithNoBias,
            neuron_tester=self.distributed_tester,
            batch_size=8,
            test_name="ddp_no_bias",
        )

    def test_ddp_with_no_init_sync(self):
        """Tests DDP functionality with `init_sync` set to False."""
        self.orchestrator.run_comparison_test(
            model_class=SimpleNet,
            neuron_tester=self.distributed_tester,
            batch_size=8,
            init_sync=False,
            test_name="ddp_no_init_sync",
        )

    def test_ddp_gradient_accumulation(self):
        """Tests DDP functionality with gradient accumulation over multiple steps."""
        self.orchestrator.run_comparison_test(
            model_class=SimpleNet,
            neuron_tester=self.distributed_tester,
            batch_size=8,
            accumulation_steps=2,
            test_name="ddp_gradient_accumulation",
        )

    @pytest.mark.xfail(
        reason="""Xfailing since we see
                       RuntimeError: Neuron does not support
                       pinned memory allocator"""
    )
    def test_ddp_unused_parameters(self):
        """Tests DDP functionality with a model containing unused parameters.

        This test uses `find_unused_parameters=True` in DDP.
        """
        self.orchestrator.run_comparison_test(
            model_class=ModelWithUnusedParams,
            neuron_tester=self.distributed_tester,
            batch_size=8,
            find_unused_parameters=True,
            test_name="ddp_unused_parameters",
        )

    @pytest.mark.xfail(
        reason="""Xfailing since we see
                    RuntimeError: Neuron does not support
                    pinned memory allocator"""
    )
    def test_ddp_static_graph(self):
        """Tests DDP functionality with `static_graph` optimization enabled."""
        self.orchestrator.run_comparison_test(
            model_class=SimpleNet,
            neuron_tester=self.distributed_tester,
            batch_size=8,
            static_graph=True,
            test_name="ddp_static_graph",
        )

    @pytest.mark.parametrize("bucket_cap_mb", [0.0001, 0.005, 0.5])
    @pytest.mark.xfail(reason="temprary xfail, due to aten linear loading")
    def test_ddp_bucket_capacity(self, bucket_cap_mb: float):
        """Tests DDP bucket capacity with multiple parameter groups.
        Tests different bucket capacity settings to verify correct bucketing behavior on Neuron

        Args:
            bucket_cap_mb: Bucket capacity in MB.
                - 0.0001 MB (0.1KB): Should force multiple buckets due to very small capacity.
                - 0.005 MB (5KB): Should force multiple buckets.
                - 0.5 MB (500KB): Should allow most parameters to fit in fewer buckets.
        """
        self.distributed_tester.run_test(run_bucket_cap_validation, bucket_cap_mb=bucket_cap_mb)
