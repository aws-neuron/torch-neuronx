"""
Test just the embedding layer with RowwiseParallel.

NOTE: this caused a problem with zero output in Neuron Torch compile

Usage:
    pytest tests/neuron_dynamo_backend/component/embedding/test_embedding.py
"""

import logging
import os
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._functional_collectives import wait_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from tests.distributed.utils import DistributedTester
from torch_neuronx.neuron_dynamo_backend import set_model_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingWrapper(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

    def init_weights(self):
        nn.init.normal_(self.tok_embeddings.weight)

    def forward(self, tokens: torch.Tensor):
        return self.tok_embeddings(tokens)


def run_embedding_test(rank, world_size, kwargs):
    """Test embedding layer with RowwiseParallel"""

    logger.debug(f"=== Rank {rank}/{world_size}: Testing Embedding Only ===")

    device = torch.device(f"neuron:{rank}")
    vocab_size = 1024
    dim = 256

    wrapper = EmbeddingWrapper(vocab_size, dim)
    wrapper.init_weights()
    logger.debug(f"Rank {rank}: Created EmbeddingWrapper")

    # Move model to neuron device BEFORE parallelization
    wrapper = wrapper.to(device)
    logger.debug(f"Rank {rank}: Moved model to {device}")

    # Apply TP with neuron device mesh
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    parallelize_module(
        wrapper,
        device_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),  # Shard along sequence dimension
            ),
        },
    )
    logger.debug(f"Rank {rank}: Applied TP with neuron device mesh")

    # Input on neuron device
    batch_size = 1
    seq_len = 16
    torch.manual_seed(42 + rank)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int32, device=device)
    logger.debug(f"Rank {rank}: Input tokens device: {tokens.device}")

    # Debug: print token range
    logger.debug(f"Rank {rank}: Token min={tokens.min().item()}, max={tokens.max().item()}")

    # Eager run on neuron
    logger.debug(f"Rank {rank}: Running eager forward...")
    with torch.no_grad():
        eager_output = wrapper(tokens)
        if hasattr(eager_output, "_local_tensor"):
            eager_output = wait_tensor(eager_output)

    eager_local = eager_output.to_local() if hasattr(eager_output, "to_local") else eager_output
    logger.debug(f"Rank {rank}: Eager output shape: {eager_local.shape}")
    logger.debug(
        f"Eager output stats - mean: {eager_local.mean():.6f}, std: {eager_local.std():.6f}"
    )

    # Compiled run on neuron
    set_model_name(f"test_minimal_embedding_rank{rank}")

    logger.debug(f"Rank {rank}: Compiling...")
    compiled_wrapper = torch.compile(wrapper, backend="neuron", fullgraph=True)

    logger.debug(f"Rank {rank}: Running compiled forward...")
    with torch.no_grad():
        compiled_output = compiled_wrapper(tokens)
        if hasattr(compiled_output, "_local_tensor"):
            compiled_output = wait_tensor(compiled_output)

    compiled_local = (
        compiled_output.to_local() if hasattr(compiled_output, "to_local") else compiled_output
    )
    logger.debug(f"Rank {rank}: Compiled output shape: {compiled_local.shape}")
    logger.debug(
        f"Rank {rank}: Compiled output stats - mean: {compiled_local.mean():.6f}, "
        f"std: {compiled_local.std():.6f}"
    )

    # Compare - both tensors are on neuron device
    abs_diff = torch.abs(eager_local - compiled_local)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    matches = torch.allclose(eager_local, compiled_local, atol=1e-3, rtol=1e-2)

    logger.debug(f"Rank {rank}: Max absolute difference: {max_abs_diff:.6e}")
    logger.debug(f"Rank {rank}: Mean absolute difference: {mean_abs_diff:.6e}")
    logger.debug(f"Rank {rank}: Outputs match: {matches}")

    dist.barrier()

    if rank == 0:
        logger.info("\n" + "=" * 80)
        if matches:
            logger.info("Embedding layer works correctly!")
        else:
            logger.info("Embedding layer produces incorrect outputs")
        logger.info("=" * 80)
    assert matches


class TestEmbedding(BaseCollectiveOpTest):
    """Test class for embedding component tests using DistributedTester."""

    @pytest.mark.multi_device
    def test_embedding_layer(self):
        """Test embedding layer with RowwiseParallel and torch.compile."""
        self.distributed_tester.run_test(run_embedding_test)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
