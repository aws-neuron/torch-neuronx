import argparse
import contextlib
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torch_neuronx

# --- Helper Functions ---


class FakeDataset(Dataset):
    """Create a random dataset for distributed training.

    Args:
        num_samples (int): Number of samples to create.
        input_dim (int): Size of each sample's final dimension.
    """

    def __init__(self, num_samples, input_dim):
        self.data = torch.randn(num_samples, input_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ToyModel(nn.Module):
    """Implement a two-layer model for distributed training."""

    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        """Compute predictions for an input batch.

        Args:
            x (torch.Tensor): Input whose final dimension has size ``10``.

        Returns:
            torch.Tensor: Predictions with the same leading dimensions as ``x`` and final
                dimension size ``5``.
        """
        return self.net2(self.relu(self.net1(x)))


def save_params_to_disk(model, rank, param_type):
    """Save gradients or weights to local disk.

    Args:
        model (nn.Module): Model whose parameters to save.
        rank (int): Process rank included in the output filename.
        param_type (str): Parameter data to save; ``"grad"`` saves available gradients,
            ``"weight"`` saves the model state, and other values save nothing.
    """
    if param_type == "grad":
        grads = {
            name: param.grad.cpu()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        torch.save(grads, f"grads_rank_{rank}.pt")
    elif param_type == "weight":
        torch.save(model.state_dict(), f"weights_rank_{rank}.pt")


def compare_params_across_ranks(model, world_size, param_type, rank):
    """Compare rank-zero saved gradients or weights with every other rank.

    Args:
        model (nn.Module): Model argument accepted but not used.
        world_size (int): Number of rank files to compare.
        param_type (str): File prefix selector; ``"grad"`` reads ``grads_rank_*.pt`` and
            ``"weight"`` reads ``weights_rank_*.pt``.
        rank (int): Process rank included in status and error messages.

    Raises:
        AssertionError: If a parameter is missing or differs from the rank-zero value.
    """
    base_filename = f"{param_type}s_rank_0.pt"
    base_params = torch.load(base_filename, map_location="cpu")

    for i in range(1, world_size):
        filename_i = f"{param_type}s_rank_{i}.pt"
        params_i = torch.load(filename_i, map_location="cpu")
        for name in base_params:
            if name in params_i:
                if not torch.equal(base_params[name], params_i[name]):
                    raise AssertionError(
                        f"Rank {rank}: {param_type.capitalize()}s for {name} DIFFER across ranks."
                    )
            else:
                raise AssertionError(
                    f"Rank {rank}: {param_type.capitalize()}s missing {name} in rank {i}."
                )

    print(f"Rank {rank}: {param_type.capitalize()}s match across all ranks.")


def cleanup_saved_files(model, world_size, rank):
    """Attempt to remove saved gradient and weight files.

    Args:
        model (nn.Module): Model argument accepted but not used.
        world_size (int): World-size argument accepted but not used.
        rank (int): Process rank included in warning messages.

    Note:
        Cleanup errors are printed as warnings and suppressed.
    """
    import glob

    try:
        for pattern in ["grads_rank_*.pt", "weights_rank_*.pt"]:
            for filename in glob.glob(pattern):
                try:
                    os.remove(filename)
                except Exception:
                    print(f"Rank {rank}: Warning: Failed to delete {filename}")
    except Exception:
        print(f"Rank {rank}: Warning: Failed to cleanup parameter files")


# --- Training Functions ---


def train_epoch(model, optimizer, dataloader, device, rank, steps_to_run=1, log_interval=10):
    """Train a model for a bounded number of steps.

    Args:
        model (nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        dataloader (DataLoader): Batches to consume.
        device (torch.device or str): Device to which each batch is moved.
        rank (int): Process rank used to restrict progress logging to rank zero.
        steps_to_run (int, optional): Number of steps to run; values below ``1`` still run
            one step. Defaults to ``1``.
        log_interval (int, optional): Number of steps between loss messages. Defaults to ``10``.

    Raises:
        ValueError: If ``dataloader`` is empty.
    """
    model.train()

    if not dataloader:
        raise ValueError(f"Rank {rank}: Dataloader is empty")

    for step, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Print loss at specified intervals (only rank 0)
        if rank == 0 and (step + 1) % log_interval == 0:
            print(f"Step {step + 1}/{steps_to_run}, Loss: {loss.item():.4f}")

        if (step + 1) >= steps_to_run:
            break


def check_and_cleanup_params(model, world_size, rank, cleanup=True):
    """Save parameters on every rank and compare them on rank zero.

    Args:
        model (nn.Module): Model whose gradients and weights to compare.
        world_size (int): Number of rank files to compare.
        rank (int): Current process rank.
        cleanup (bool, optional): On rank zero, remove saved files after comparison. Defaults
            to ``True``.

    Note:
        Errors are printed and suppressed.
    """
    try:
        save_params_to_disk(model, rank, "grad")
        save_params_to_disk(model, rank, "weight")

        dist.barrier()

        if rank == 0:
            compare_params_across_ranks(model, world_size, "grad", rank)
            compare_params_across_ranks(model, world_size, "weight", rank)

            if cleanup:
                cleanup_saved_files(model, world_size, rank)

    except Exception as e:
        print(f"Rank {rank}: Error in parameter checking: {e}")


def parse_args():
    """Parse command-line arguments for the distributed training example.

    Returns:
        argparse.Namespace: Parsed training and parameter-comparison options.
    """
    parser = argparse.ArgumentParser(description="Multi-node data parallel training example")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of samples in dataset")
    parser.add_argument("--input-dim", type=int, default=10, help="Input dimension")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per process")
    parser.add_argument("--steps", type=int, default=1, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Steps interval for logging loss"
    )
    parser.add_argument(
        "--compare-params",
        action="store_true",
        default=True,
        help="Compare gradients and weights across ranks (default: True)",
    )
    parser.add_argument(
        "--no-compare-params",
        action="store_false",
        dest="compare_params",
        help="Disable parameter comparison",
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Cleanup parameter files after comparison"
    )
    return parser.parse_args()


# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()
    try:
        dist.init_process_group(
            backend="neuron", world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"])
        )
    except KeyError as e:
        print(f"Error initializing process group: {e}. Ensure WORLD_SIZE and RANK are set.")
        exit()

    rank = int(os.environ["RANK"])
    world_size = dist.get_world_size()

    # Set random seeds for reproducibility
    base_seed = 1234
    seed = base_seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Rank {rank}/{world_size - 1} initialized with seed {seed}.")

    try:
        neuron_device = f"neuron:{torch_neuronx.current_device()}"
        print(f"Using Neuron device: {neuron_device}")
    except Exception as e:
        print(f"Error getting Neuron device: {e}")
        exit()

    model = ToyModel().to(neuron_device)
    ddp_model = DistributedDataParallel(
        model, init_sync=True, device_ids=[torch_neuronx.current_device()]
    )
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr)

    num_samples = args.num_samples
    input_dim = args.input_dim
    batch_size = args.batch_size

    dataset = FakeDataset(num_samples, input_dim)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    try:
        # Configure training parameters
        steps_to_run = args.steps

        # Perform training steps
        print(f"Rank {rank}: Starting training for {steps_to_run} step(s)...")
        train_epoch(
            ddp_model, optimizer, dataloader, neuron_device, rank, steps_to_run, args.log_interval
        )
        print(f"Rank {rank}: Training completed.")

        # Check gradients and weights after training if enabled
        if args.compare_params:
            print(f"Rank {rank}: Comparing parameters across ranks...")
            check_and_cleanup_params(ddp_model, world_size, rank, cleanup=args.cleanup)
        else:
            print(f"Rank {rank}: Skipping parameter comparison.")

        if rank == 0:
            print(f"Rank {rank}: Script finished.")
    finally:
        # Only rank 0 cleans up all files to avoid race conditions
        if rank == 0:
            cleanup_saved_files(ddp_model, world_size, rank)
        with contextlib.suppress(Exception):
            dist.destroy_process_group()
    exit(0)
