"""Modular DDP testing framework for CPU and Neuron comparison."""

import copy
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel

from ..utils import get_free_port


def _cpu_test_wrapper(rank: int, world_size: int, test_fn: Any, kwargs: dict[str, Any]) -> None:
    """Wrapper function for CPU distributed testing."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # Port is already set by the parent process
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        test_fn(rank, world_size, kwargs)
    finally:
        dist.destroy_process_group()


def _run_cpu_ddp_test(rank: int, world_size: int, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Run CPU DDP test - picklable function."""
    runner = DDPRunner("cpu")
    model_class = kwargs.pop("model_class")
    result = runner.run_test(rank, world_size, kwargs, model_class)
    return result


def _run_neuron_ddp_test(
    rank: int, world_size: int, kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run Neuron DDP test - picklable function."""
    runner = DDPRunner("neuron")
    model_class = kwargs.pop("model_class")
    result = runner.run_test(rank, world_size, kwargs, model_class)
    return result


class CPUDistributedTester:
    """CPU-only distributed tester without Neuron dependencies."""

    def __init__(self, world_size: int):
        self.world_size = world_size

    def run_test(self, test_fn: Any, **kwargs: Any) -> None:
        """Run distributed test using torch.multiprocessing.spawn."""
        import torch.multiprocessing as mp

        os.environ["MASTER_PORT"] = get_free_port()
        mp.spawn(
            _cpu_test_wrapper,
            args=(self.world_size, test_fn, kwargs),
            nprocs=self.world_size,
            join=True,
        )


class DDPRunner:
    """Unified DDP test runner for both CPU and Neuron devices.

    Args:
        device (str): The device to run the tests on, either "cpu" or "neuron".
    """

    def __init__(self, device: str):
        self.device = device

    def create_dataset(self, batch_size: int, input_size: int, world_size: int, rank: int):
        """Create a synthetic dataset and a DistributedSampler-wrapped DataLoader.

        Args:
            batch_size (int): The total batch size for the dataset.
            input_size (int): The size of the input features.
            world_size (int): The total number of distributed processes.
            rank (int): The current process's rank.

        Returns:
            torch.utils.data.DataLoader: A DataLoader configured for distributed training.
        """
        torch.manual_seed(42)
        total_samples = batch_size * 2 * 5
        dataset = torch.utils.data.TensorDataset(
            torch.randn(total_samples, input_size), torch.randn(total_samples, 5)
        )
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size // world_size, sampler=sampler
        )

    def save_results(
        self, results: list[dict[str, Any]], rank: int, test_name: str = "default"
    ) -> None:
        """Save the training results for a given rank to a file.

        Args:
            results (list): A list of dictionaries containing the results for each training step.
            rank (int): The rank of the current process.
            test_name (str): Name of the test to prevent overwrites.
        """
        torch.save(results, f"./{self.device}_{test_name}_results_rank_{rank}.pt")

    def training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_input: torch.Tensor,
        batch_target: torch.Tensor,
        step_idx: int,
        rank: int,
        accumulation_steps: int,
        test_name: str = "default",
    ) -> dict[str, Any]:
        """Execute a single training step, including forward pass, loss calculation,
        backward pass, and optimizer step.

        Args:
            model (torch.nn.Module): The DDP-wrapped model.
            optimizer (torch.optim.Optimizer): The optimizer for model parameters.
            batch_input (torch.Tensor): The input tensor for the current batch.
            batch_target (torch.Tensor): The target tensor for the current batch.
            step_idx (int): The current training step index.
            rank (int): The rank of the current process.
            accumulation_steps (int): Number of steps to accumulate gradients before optimizing.

        Returns:
            dict[str, Any]: A dictionary containing the detached CPU output and the loss item.
        """
        optimizer.zero_grad()

        # Save inputs and weights before forward pass

        batch_input = batch_input.to(self.device)
        batch_target = batch_target.to(self.device)

        output = model(batch_input)
        loss = functional.mse_loss(output, batch_target)
        loss.backward()
        if step_idx % accumulation_steps == 0:
            optimizer.step()

        return {"output": output.detach().cpu(), "loss": loss.item()}

    def run_test(
        self, rank: int, world_size: int, kwargs: dict[str, Any], model_class: type
    ) -> list[dict[str, Any]]:
        """Run the DDP test for the specified device (CPU or Neuron).

        This method initializes the model, DDP wrapper, dataloader, and optimizer,
        then executes the training loop.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of distributed processes.
            kwargs (dict): Additional keyword arguments for test configuration,
                           e.g., batch_size, input_size, find_unused_parameters.
            model_class (torch.nn.Module): The class of the model to be tested.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the results for each
                training step.
        """
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True, warn_only=True)

        batch_size = kwargs.get("batch_size", 8)
        input_size = kwargs.get("input_size", 10)
        test_name = kwargs.get("test_name", "default")

        # Create model with DDP
        if self.device == "neuron":
            cpu_model = model_class(input_size)
            model = copy.deepcopy(cpu_model).to("neuron")
        else:
            model = model_class(input_size).to(self.device)

        extra_args = {}
        if "find_unused_parameters" in kwargs:
            extra_args["find_unused_parameters"] = kwargs["find_unused_parameters"]
        elif "static_graph" in kwargs:
            extra_args["static_graph"] = kwargs["static_graph"]
        elif "init_sync" in kwargs:
            extra_args["init_sync"] = kwargs["init_sync"]

        ddp_model = DistributedDataParallel(model, **extra_args)

        # Create dataloader
        dataloader = self.create_dataset(batch_size, input_size, world_size, rank)

        # Optimizer
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

        # Training loop
        results = []

        accumulation_steps = 1
        if "accumulation_steps" in kwargs:
            accumulation_steps = kwargs["accumulation_steps"]

        dataloader_list = list(dataloader)
        for i, (batch_input, batch_target) in enumerate(dataloader_list):
            result = self.training_step(
                ddp_model,
                optimizer,
                batch_input,
                batch_target,
                i,
                rank,
                accumulation_steps,
                test_name,
            )
            results.append(result)

        # Save weights after final step
        weights_dict = {name: param.detach().cpu() for name, param in ddp_model.named_parameters()}
        torch.save(
            {"weights": weights_dict},
            f"./{self.device}_{test_name}_final_weights_rank_{rank}.pt",
        )

        self.save_results(results, rank, test_name)
        return results


class ResultCollector:
    """Collects and compares results from distributed tests."""

    @staticmethod
    def collect_results(
        prefix: str, world_size: int, test_name: str = "default"
    ) -> dict[int, list[dict[str, Any]]]:
        """Collects results from files saved by DDPRunner for each rank.

        Args:
            prefix (str): The prefix used for saving result files (e.g., "cpu", "neuron").
            world_size (int): The total number of distributed processes.
            test_name (str): Name of the test to match saved files.

        Returns:
            dict[int, list[dict[str, Any]]]: A dictionary where keys are ranks and values
                are lists of results for that rank.
        """
        results = {}
        for rank in range(world_size):
            results[rank] = torch.load(f"./{prefix}_{test_name}_results_rank_{rank}.pt")
            os.remove(f"./{prefix}_{test_name}_results_rank_{rank}.pt")
            # Clean up final weight files
            weight_file = f"./{prefix}_{test_name}_final_weights_rank_{rank}.pt"
            if os.path.exists(weight_file):
                os.remove(weight_file)
        return results

    @staticmethod
    def compare_results(
        cpu_results: dict[int, list[dict[str, Any]]],
        neuron_results: dict[int, list[dict[str, Any]]],
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> None:
        """Compares the collected CPU and Neuron results for numerical equivalence.

        Args:
            cpu_results (dict): Results collected from CPU DDP tests.
            neuron_results (dict): Results collected from Neuron DDP tests.
            rtol (float): Relative tolerance for `torch.allclose`.
            atol (float): Absolute tolerance for `torch.allclose`.

        Raises:
            AssertionError: If the number of batches differs or if outputs/losses mismatch.
        """
        for rank in range(len(cpu_results)):
            if rank in cpu_results and rank in neuron_results:
                cpu_rank_results = cpu_results[rank]
                neuron_rank_results = neuron_results[rank]

                assert len(cpu_rank_results) == len(
                    neuron_rank_results
                ), f"Different number of batches for rank {rank}"

                for i, (cpu_batch, neuron_batch) in enumerate(
                    zip(cpu_rank_results, neuron_rank_results, strict=False)
                ):
                    cpu_output = cpu_batch["output"]
                    neuron_output = neuron_batch["output"]

                    assert torch.allclose(cpu_output, neuron_output, rtol=1e-5, atol=1e-5), (
                        f"Output mismatch at rank {rank}, batch {i}, "
                        f"max_diff={torch.abs(cpu_output - neuron_output).max():.2e}"
                    )

                    assert torch.allclose(
                        torch.tensor(cpu_batch["loss"]),
                        torch.tensor(neuron_batch["loss"]),
                        rtol=1e-5,
                        atol=1e-5,
                    ), (
                        f"Loss mismatch at rank {rank}, batch {i}, "
                        f"{cpu_batch['loss']=}, {neuron_batch['loss']=}"
                    )


class DDPTestOrchestrator:
    """Orchestrates CPU vs Neuron DDP comparison tests.

    This class manages the execution of distributed tests on both CPU and Neuron
    devices, collects their results, and performs a comparison to ensure correctness.

    Args:
        world_size (int): The total number of distributed processes to use for the
            tests. Defaults to 2.
    """

    def __init__(self, world_size: int = 2):
        self.world_size = world_size
        self.cpu_tester = CPUDistributedTester(world_size)
        self.cpu_runner = DDPRunner("cpu")
        self.neuron_runner = DDPRunner("neuron")
        self.collector = ResultCollector()

    def run_comparison_test(
        self, model_class: type, neuron_tester: Any, **kwargs: Any
    ) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
        """Runs a complete comparison test between CPU and Neuron DDP execution.

        This involves running the test on CPU, then on Neuron, collecting results
        from both, and finally comparing them.

        Args:
            model_class (torch.nn.Module): The class of the model to be tested.
            neuron_tester: An instance of a Neuron-specific distributed tester.
            **kwargs: Additional keyword arguments to pass to the DDPRunner's run_test method.

        Returns:
            tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]: A tuple
                containing two dictionaries: (cpu_results, neuron_results).
        """
        # Add model_class to kwargs for the test functions
        test_kwargs = kwargs.copy()
        test_kwargs["model_class"] = model_class
        test_name = kwargs.get("test_name", "default")

        # Run CPU baseline
        self.cpu_tester.run_test(_run_cpu_ddp_test, **test_kwargs)

        # Run Neuron DDP
        neuron_tester.run_test(_run_neuron_ddp_test, **test_kwargs)

        # Collect and compare results
        cpu_results = self.collector.collect_results("cpu", self.world_size, test_name)
        neuron_results = self.collector.collect_results("neuron", self.world_size, test_name)

        self.collector.compare_results(cpu_results, neuron_results)

        return cpu_results, neuron_results
