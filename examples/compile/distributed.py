"""
Distributed Example

Multi-process execution with torchrun using functional collectives.

Run: torchrun --nproc_per_node=2 distributed.py
"""

import torch
import torch.distributed as dist

# torch.compile requires the use of functional collectives
from torch.distributed._functional_collectives import all_reduce


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x):
        x = self.linear(x)
        x = all_reduce(x, "sum", list(range(dist.get_world_size())))
        return x


def main():
    # Initialize distributed backend
    dist.init_process_group("neuron")
    rank = dist.get_rank()
    torch.neuron.set_device(rank)

    device = f"neuron:{rank}"

    model = SimpleModel().to(device)
    compiled_model = torch.compile(model, backend="neuron")

    with torch.inference_mode():
        x = torch.randn(4, 64).to(device)
        y = compiled_model(x)

    print(f"[Rank {rank}] Output shape: {y.shape}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
